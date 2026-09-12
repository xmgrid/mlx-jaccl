use crate::config::Config;
use anyhow::{Context, Result};
use std::fs;
use std::path::PathBuf;
use std::process::Command;

const AGENT_LABEL: &str = "com.mlxjaccl.agent";
const CONTROLLER_LABEL: &str = "com.mlxjaccl.controller";
const NETWORK_LABEL: &str = "com.mlxjaccl.network";

pub fn install(cfg: &Config, include_controller: bool, sudo_password: Option<&str>) -> Result<()> {
    let bin_dir = cfg.cluster_dir.join("bin");
    fs::create_dir_all(&bin_dir)?;
    fs::create_dir_all(cfg.cluster_dir.join("libexec"))?;
    fs::create_dir_all(&cfg.log_dir)?;
    fs::create_dir_all(cfg.ui_dir())?;

    let dest_bin = bin_dir.parent().unwrap_or(&bin_dir).join("libexec/mlxctl");
    let current = Config::binary_path();
    if current != dest_bin {
        fs::copy(&current, &dest_bin)
            .with_context(|| format!("copy binary to {}", dest_bin.display()))?;
        let _ = Command::new("chmod")
            .args(["+x", dest_bin.to_str().unwrap()])
            .status();
    }

    let wrapper = bin_dir.join("mlx-net-up");
    fs::write(
        &wrapper,
        format!(
            "#!/bin/bash\nexec {:?} net-up --config {:?}\n",
            dest_bin,
            cfg.cluster_dir.join("mlxctl.toml")
        ),
    )?;
    let _ = Command::new("chmod")
        .args(["+x", wrapper.to_str().unwrap()])
        .status();

    let Some(pw) = sudo_password else {
        anyhow::bail!("install needs --sudo-password to write LaunchDaemons and sudoers");
    };

    // Drop any previous system-domain copies; those cannot talk to the LAN on macOS 26.
    for label in [AGENT_LABEL, CONTROLLER_LABEL] {
        sudo_pw(pw, &["launchctl", "bootout", &format!("system/{label}")]);
        sudo_pw(
            pw,
            &["rm", "-f", &format!("/Library/LaunchDaemons/{label}.plist")],
        );
    }

    let sudoers = format!(
        "{} ALL=(root) NOPASSWD: {}\n",
        crate::config::username(),
        wrapper.display()
    );
    let tmp_sudo = cfg.cluster_dir.join(".mlxjaccl.sudoers");
    fs::write(&tmp_sudo, sudoers)?;
    sudo_pw(pw, &["cp", tmp_sudo.to_str().unwrap(), "/etc/sudoers.d/mlxjaccl"]);
    sudo_pw(pw, &["chmod", "440", "/etc/sudoers.d/mlxjaccl"]);
    sudo_pw(pw, &["chown", "root:wheel", "/etc/sudoers.d/mlxjaccl"]);
    let _ = fs::remove_file(&tmp_sudo);

    install_root_daemon(pw, cfg, &wrapper)?;

    let uid = user_uid();
    let home = crate::config::home_dir();
    let agents = home.join("Library/LaunchAgents");
    fs::create_dir_all(&agents)?;

    let agent_plist = agents.join(format!("{AGENT_LABEL}.plist"));
    fs::write(
        &agent_plist,
        user_plist(
            AGENT_LABEL,
            &dest_bin,
            cfg,
            "agent",
            &cfg.log_dir.join("agent.out.log"),
            &cfg.log_dir.join("agent.err.log"),
        ),
    )?;
    reload_user(uid, AGENT_LABEL, &agent_plist, pw)?;

    if include_controller {
        let ctl_plist = agents.join(format!("{CONTROLLER_LABEL}.plist"));
        fs::write(
            &ctl_plist,
            user_plist(
                CONTROLLER_LABEL,
                &dest_bin,
                cfg,
                "controller",
                &cfg.log_dir.join("controller.out.log"),
                &cfg.log_dir.join("controller.err.log"),
            ),
        )?;
        reload_user(uid, CONTROLLER_LABEL, &ctl_plist, pw)?;
    }
    Ok(())
}

pub fn uninstall(cfg: &Config, sudo_password: Option<&str>) -> Result<()> {
    let uid = user_uid();
    let home = crate::config::home_dir();
    let agents = home.join("Library/LaunchAgents");
    for label in [AGENT_LABEL, CONTROLLER_LABEL] {
        let _ = Command::new("launchctl")
            .args(["bootout", &format!("gui/{uid}/{label}")])
            .status();
        let _ = fs::remove_file(agents.join(format!("{label}.plist")));
        if let Some(pw) = sudo_password {
            sudo_pw(pw, &["launchctl", "bootout", &format!("system/{label}")]);
            sudo_pw(
                pw,
                &["rm", "-f", &format!("/Library/LaunchDaemons/{label}.plist")],
            );
        }
    }
    if let Some(pw) = sudo_password {
        sudo_pw(pw, &["launchctl", "bootout", &format!("system/{NETWORK_LABEL}")]);
        sudo_pw(
            pw,
            &[
                "rm",
                "-f",
                &format!("/Library/LaunchDaemons/{NETWORK_LABEL}.plist"),
            ],
        );
        sudo_pw(pw, &["rm", "-f", "/etc/sudoers.d/mlxjaccl"]);
    }
    let _ = cfg;
    Ok(())
}

fn install_root_daemon(pw: &str, cfg: &Config, wrapper: &PathBuf) -> Result<()> {
    let tmp = cfg.cluster_dir.join(format!(".{NETWORK_LABEL}.plist"));
    let dest = format!("/Library/LaunchDaemons/{NETWORK_LABEL}.plist");
    fs::write(&tmp, root_plist(NETWORK_LABEL, wrapper, cfg))?;
    sudo_pw(pw, &["cp", tmp.to_str().unwrap(), &dest]);
    sudo_pw(pw, &["chown", "root:wheel", &dest]);
    sudo_pw(pw, &["chmod", "644", &dest]);
    sudo_pw(pw, &["launchctl", "bootout", &format!("system/{NETWORK_LABEL}")]);
    if !sudo_pw(pw, &["launchctl", "bootstrap", "system", &dest]) {
        tracing::warn!("launchctl bootstrap {NETWORK_LABEL} failed");
    }
    sudo_pw(
        pw,
        &[
            "launchctl",
            "kickstart",
            "-k",
            &format!("system/{NETWORK_LABEL}"),
        ],
    );
    let _ = fs::remove_file(tmp);
    Ok(())
}

fn user_plist(
    label: &str,
    bin: &PathBuf,
    cfg: &Config,
    subcmd: &str,
    stdout: &PathBuf,
    stderr: &PathBuf,
) -> String {
    let toml = cfg.cluster_dir.join("mlxctl.toml");
    let home = crate::config::home_dir();
    let path_env = format!(
        "{}/.local/bin:/opt/homebrew/bin:/usr/bin:/bin:/usr/sbin:/sbin",
        home.display()
    );
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{}</string>
    <string>{subcmd}</string>
    <string>--config</string>
    <string>{}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>KeepAlive</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>{}</string>
  <key>StandardOutPath</key>
  <string>{}</string>
  <key>StandardErrorPath</key>
  <string>{}</string>
  <key>EnvironmentVariables</key>
  <dict>
    <key>HOME</key>
    <string>{}</string>
    <key>PATH</key>
    <string>{path_env}</string>
  </dict>
</dict>
</plist>
"#,
        bin.display(),
        toml.display(),
        cfg.cluster_dir.display(),
        stdout.display(),
        stderr.display(),
        home.display()
    )
}

fn root_plist(label: &str, wrapper: &PathBuf, cfg: &Config) -> String {
    format!(
        r#"<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>Label</key>
  <string>{label}</string>
  <key>ProgramArguments</key>
  <array>
    <string>{}</string>
  </array>
  <key>RunAtLoad</key>
  <true/>
  <key>WorkingDirectory</key>
  <string>{}</string>
  <key>StandardOutPath</key>
  <string>{}</string>
  <key>StandardErrorPath</key>
  <string>{}</string>
</dict>
</plist>
"#,
        wrapper.display(),
        cfg.cluster_dir.display(),
        cfg.log_dir.join("network.out.log").display(),
        cfg.log_dir.join("network.err.log").display()
    )
}

fn reload_user(uid: u32, label: &str, plist: &PathBuf, sudo_password: &str) -> Result<()> {
    let domain = format!("gui/{uid}");
    let target = format!("{domain}/{label}");
    sudo_pw(sudo_password, &["launchctl", "bootout", &target]);
    if !sudo_pw(
        sudo_password,
        &["launchctl", "bootstrap", &domain, plist.to_str().unwrap()],
    ) {
        tracing::warn!("sudo launchctl bootstrap {label} failed");
        let _ = Command::new("launchctl")
            .args(["bootstrap", &domain, plist.to_str().unwrap()])
            .status();
    }
    sudo_pw(sudo_password, &["launchctl", "kickstart", "-k", &target]);
    Ok(())
}

fn user_uid() -> u32 {
    Command::new("id")
        .arg("-u")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .and_then(|s| s.trim().parse().ok())
        .unwrap_or(501)
}

fn sudo_pw(password: &str, args: &[&str]) -> bool {
    let child = Command::new("/usr/bin/sudo")
        .arg("-S")
        .args(args)
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn();
    let Ok(mut child) = child else {
        return false;
    };
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        let _ = writeln!(stdin, "{password}");
    }
    child.wait().map(|s| s.success()).unwrap_or(false)
}
