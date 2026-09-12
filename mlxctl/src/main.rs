mod agent;
mod anthropic;
mod config;
mod controller;
mod launchd;
mod models;
mod network;
mod proxy;
mod responses;
mod serve;
mod stack;
mod state;

use anyhow::Result;
use clap::{Parser, Subcommand};
use config::Config;
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "mlxctl", about = "MLX-JACCL cluster control plane")]
struct Cli {
    #[arg(long, global = true)]
    config: Option<PathBuf>,
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Node agent (every Mac Studio)
    Agent,
    /// Control plane + UI (rank 0)
    Controller,
    /// Apply Thunderbolt mesh IPs (run as root)
    NetUp,
    /// Install launchd services so this machine comes back after reboot
    Install {
        #[arg(long)]
        controller: bool,
        #[arg(long)]
        sudo_password: Option<String>,
    },
    Uninstall {
        #[arg(long)]
        sudo_password: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "mlxctl=info,tower_http=info".into()),
        )
        .init();

    let cli = Cli::parse();
    let config_path = cli
        .config
        .unwrap_or_else(|| config::home_dir().join("mlx-cluster/mlxctl.toml"));
    let cfg = Config::load(&config_path)?;

    match cli.cmd {
        Cmd::Agent => agent::run(cfg).await?,
        Cmd::Controller => controller::run(cfg).await?,
        Cmd::NetUp => {
            let st = network::apply_this_node(&cfg)?;
            println!("{}", serde_json::to_string_pretty(&st)?);
        }
        Cmd::Install {
            controller,
            sudo_password,
        } => {
            launchd::install(&cfg, controller, sudo_password.as_deref())?;
            println!("installed (controller={controller})");
        }
        Cmd::Uninstall { sudo_password } => {
            launchd::uninstall(&cfg, sudo_password.as_deref())?;
            println!("uninstalled");
        }
    }
    Ok(())
}
