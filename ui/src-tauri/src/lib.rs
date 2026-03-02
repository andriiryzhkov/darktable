mod commands;
mod ipc_client;
mod server_manager;
mod shm_reader;

use std::collections::HashMap;
use std::sync::atomic::AtomicBool;
use std::sync::Mutex;

pub struct AppState {
    pub server: Mutex<Option<server_manager::ServerProcess>>,
    pub client: tokio::sync::Mutex<Option<ipc_client::IpcClient>>,
    pub shm_handles: Mutex<HashMap<String, (shm_reader::ShmHandle, shm_reader::ShmHandle)>>,
    pub starting: AtomicBool,
}

pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_shell::init())
        .manage(AppState {
            server: Mutex::new(None),
            client: tokio::sync::Mutex::new(None),
            shm_handles: Mutex::new(HashMap::new()),
            starting: AtomicBool::new(false),
        })
        .invoke_handler(tauri::generate_handler![
            commands::start_server,
            commands::ping,
            commands::catalog_query,
            commands::catalog_get_thumbnail,
            commands::develop_open,
            commands::develop_close,
            commands::develop_set_params,
            commands::develop_request_preview,
            commands::get_preview_frame,
        ])
        .run(tauri::generate_context!())
        .expect("error running darktable-ui");
}
