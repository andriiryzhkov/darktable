use std::sync::atomic::Ordering;

use tauri::State;

use crate::ipc_client::IpcClient;
use crate::server_manager::ServerProcess;
use crate::shm_reader::ShmHandle;
use crate::AppState;

#[tauri::command]
pub async fn start_server(state: State<'_, AppState>) -> Result<String, String> {
    // Prevent double-spawning (React StrictMode calls effects twice)
    if state
        .starting
        .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
        .is_err()
    {
        return Err("server start already in progress".to_string());
    }

    // Check if already running
    {
        let guard = state.server.lock().map_err(|e| e.to_string())?;
        if guard.is_some() {
            state.starting.store(false, Ordering::SeqCst);
            return Err("server already running".to_string());
        }
    }

    // Find server binary relative to the repo build directory
    let server_bin = std::env::var("DT_SERVER_BIN").unwrap_or_else(|_| {
        // Default: look in ../build/bin/ relative to ui/
        let mut path = std::env::current_dir().unwrap_or_default();
        // We might be in ui/ or ui/src-tauri/, go up to repo root
        while path.file_name().map_or(false, |n| n == "ui" || n == "src-tauri") {
            path.pop();
        }
        path.join("build/bin/darktable-server")
            .to_string_lossy()
            .to_string()
    });

    let configdir = std::env::var("DT_CONFIGDIR")
        .unwrap_or_else(|_| {
            let home = std::env::var("HOME").unwrap_or_default();
            format!("{home}/.config/darktable-tauri-test")
        });

    eprintln!("[tauri] starting server: {server_bin}");
    eprintln!("[tauri] configdir: {configdir}");

    let server = ServerProcess::start(&server_bin, &configdir)?;
    let socket_path = server.socket_path.clone();

    // Connect IPC client
    let client = IpcClient::connect(&socket_path).await?;

    // Store in state
    {
        let mut guard = state.server.lock().map_err(|e| e.to_string())?;
        *guard = Some(server);
    }
    {
        let mut guard = state.client.lock().await;
        *guard = Some(client);
    }

    Ok(socket_path)
}

#[tauri::command]
pub async fn ping(state: State<'_, AppState>) -> Result<serde_json::Value, String> {
    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request("system.ping", serde_json::json!({}))
        .await
}

#[tauri::command]
pub async fn catalog_query(
    state: State<'_, AppState>,
    offset: i64,
    limit: i64,
) -> Result<serde_json::Value, String> {
    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request(
            "catalog.query",
            serde_json::json!({"offset": offset, "limit": limit}),
        )
        .await
}

#[tauri::command]
pub async fn catalog_get_thumbnail(
    state: State<'_, AppState>,
    imgid: i64,
) -> Result<serde_json::Value, String> {
    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request(
            "catalog.get_thumbnail",
            serde_json::json!({"imgid": imgid}),
        )
        .await
}

#[tauri::command]
pub async fn develop_open(
    state: State<'_, AppState>,
    imgid: i64,
    width: i64,
    height: i64,
) -> Result<serde_json::Value, String> {
    let result = {
        let mut guard = state.client.lock().await;
        let client = guard.as_mut().ok_or("not connected")?;
        client
            .request(
                "develop.open",
                serde_json::json!({"imgid": imgid, "width": width, "height": height}),
            )
            .await?
    };

    // Open SHM handles for both buffers
    let session_id = result
        .get("session_id")
        .and_then(|v| v.as_str())
        .ok_or("missing session_id in response")?
        .to_string();

    let shm_names = result
        .get("shm_names")
        .and_then(|v| v.as_array())
        .ok_or("missing shm_names in response")?;

    let name0 = shm_names
        .first()
        .and_then(|v| v.as_str())
        .ok_or("missing shm_names[0]")?;
    let name1 = shm_names
        .get(1)
        .and_then(|v| v.as_str())
        .ok_or("missing shm_names[1]")?;

    let pw = result
        .get("preview_width")
        .and_then(|v| v.as_u64())
        .unwrap_or(width as u64) as u32;
    let ph = result
        .get("preview_height")
        .and_then(|v| v.as_u64())
        .unwrap_or(height as u64) as u32;

    let shm0 = ShmHandle::open(name0, pw, ph)?;
    let shm1 = ShmHandle::open(name1, pw, ph)?;

    {
        let mut shm_map = state.shm_handles.lock().map_err(|e| e.to_string())?;
        shm_map.insert(session_id, (shm0, shm1));
    }

    Ok(result)
}

#[tauri::command]
pub async fn develop_close(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<serde_json::Value, String> {
    // Remove SHM handles
    {
        let mut shm_map = state.shm_handles.lock().map_err(|e| e.to_string())?;
        shm_map.remove(&session_id);
    }

    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request(
            "develop.close",
            serde_json::json!({"session_id": session_id}),
        )
        .await
}

#[tauri::command]
pub async fn develop_set_params(
    state: State<'_, AppState>,
    session_id: String,
    op: String,
    params: serde_json::Value,
) -> Result<serde_json::Value, String> {
    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request(
            "develop.set_params",
            serde_json::json!({
                "session_id": session_id,
                "op": op,
                "params": params,
            }),
        )
        .await
}

#[tauri::command]
pub async fn develop_request_preview(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<serde_json::Value, String> {
    let mut guard = state.client.lock().await;
    let client = guard.as_mut().ok_or("not connected")?;
    client
        .request(
            "develop.request_preview",
            serde_json::json!({"session_id": session_id}),
        )
        .await
}

#[tauri::command]
pub async fn get_preview_frame(
    state: State<'_, AppState>,
    session_id: String,
    front_buffer: usize,
) -> Result<Vec<u8>, String> {
    let shm_map = state.shm_handles.lock().map_err(|e| e.to_string())?;
    let (shm0, shm1) = shm_map
        .get(&session_id)
        .ok_or("no SHM handles for session")?;

    let shm = if front_buffer == 0 { shm0 } else { shm1 };
    let frame = shm
        .read_frame()
        .ok_or("frame not ready or invalid header")?;

    Ok(frame.pixels)
}
