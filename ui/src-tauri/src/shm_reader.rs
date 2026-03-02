use std::ffi::CString;

const SHM_MAGIC: u32 = 0x44545348; // "DTSH"
const SHM_VERSION: u32 = 1;
const SHM_HEADER_SIZE: usize = 64;

#[repr(C)]
struct ShmHeader {
    magic: u32,
    version: u32,
    width: u32,
    height: u32,
    stride: u32,
    format: u32,
    sequence: u64,
    ready: u32,
    _reserved: [u32; 5],
}

pub struct ShmHandle {
    ptr: *mut u8,
    size: usize,
    name: String,
}

// Safety: ShmHandle is only accessed from the Tauri command thread context
// and the underlying mmap is read-only from our side.
unsafe impl Send for ShmHandle {}
unsafe impl Sync for ShmHandle {}

pub struct FrameData {
    pub width: u32,
    pub height: u32,
    pub sequence: u64,
    pub pixels: Vec<u8>,
}

impl ShmHandle {
    /// Open a POSIX shared memory segment by name.
    pub fn open(name: &str, width: u32, height: u32) -> Result<Self, String> {
        let expected_size = SHM_HEADER_SIZE + (width as usize) * (height as usize) * 4;

        let c_name =
            CString::new(name).map_err(|_| "invalid SHM name".to_string())?;

        unsafe {
            let fd = libc::shm_open(c_name.as_ptr(), libc::O_RDONLY, 0);
            if fd < 0 {
                return Err(format!(
                    "shm_open({name}) failed: {}",
                    std::io::Error::last_os_error()
                ));
            }

            let ptr = libc::mmap(
                std::ptr::null_mut(),
                expected_size,
                libc::PROT_READ,
                libc::MAP_SHARED,
                fd,
                0,
            );

            libc::close(fd);

            if ptr == libc::MAP_FAILED {
                return Err(format!(
                    "mmap failed for {name}: {}",
                    std::io::Error::last_os_error()
                ));
            }

            Ok(Self {
                ptr: ptr as *mut u8,
                size: expected_size,
                name: name.to_string(),
            })
        }
    }

    /// Read a frame from the SHM buffer if it's ready.
    pub fn read_frame(&self) -> Option<FrameData> {
        unsafe {
            let header = &*(self.ptr as *const ShmHeader);

            // Validate header
            if header.magic != SHM_MAGIC || header.version != SHM_VERSION {
                return None;
            }

            // Check if the buffer is ready to read
            let ready =
                std::sync::atomic::AtomicU32::from_ptr(&header.ready as *const u32 as *mut u32)
                    .load(std::sync::atomic::Ordering::Acquire);
            if ready != 1 {
                return None;
            }

            let width = header.width;
            let height = header.height;
            let sequence = header.sequence;
            let pixel_size = (width as usize) * (height as usize) * 4;

            // Copy pixel data from offset 64
            let pixel_ptr = self.ptr.add(SHM_HEADER_SIZE);
            let mut pixels = vec![0u8; pixel_size];
            std::ptr::copy_nonoverlapping(pixel_ptr, pixels.as_mut_ptr(), pixel_size);

            Some(FrameData {
                width,
                height,
                sequence,
                pixels,
            })
        }
    }
}

impl Drop for ShmHandle {
    fn drop(&mut self) {
        unsafe {
            libc::munmap(self.ptr as *mut libc::c_void, self.size);
        }
        // We do NOT shm_unlink — the server owns the SHM lifecycle.
        eprintln!("[shm] unmapped {}", self.name);
    }
}
