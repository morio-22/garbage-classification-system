"""桌面版启动器：启动 Streamlit 服务并自动打开浏览器。"""

from __future__ import annotations

import ctypes
import os
import socket
import sys
import threading
import time
import webbrowser
from pathlib import Path


APP_NAME = "垃圾分类智能识别系统"
MODEL_RELATIVE_PATH = Path("models") / "garbage_resnet18.pth"


def get_resource_roots() -> list[Path]:
    """返回源码运行和 PyInstaller 打包运行时可能存放资源的目录。"""

    roots: list[Path] = []

    # PyInstaller 打包后，数据文件通常会放在 sys._MEIPASS 指向的目录。
    bundled_root = getattr(sys, "_MEIPASS", None)
    if bundled_root:
        roots.append(Path(bundled_root).resolve())

    if getattr(sys, "frozen", False):
        roots.append(Path(sys.executable).resolve().parent)
    else:
        roots.append(Path(__file__).resolve().parent)

    roots.append(Path.cwd().resolve())
    return list(dict.fromkeys(roots))


def find_resource(relative_path: Path) -> Path | None:
    """在可能的资源目录里查找某个文件。"""

    for root in get_resource_roots():
        candidate = root / relative_path
        if candidate.exists():
            return candidate
    return None


def show_error(message: str) -> None:
    """在控制台和 Windows 弹窗中显示错误信息。"""

    print(message)
    if os.name == "nt":
        ctypes.windll.user32.MessageBoxW(None, message, APP_NAME, 0x10)


def find_available_port(start_port: int = 8501, end_port: int = 8599) -> int:
    """寻找一个可用端口，避免和已经打开的 Streamlit 服务冲突。"""

    for port in range(start_port, end_port + 1):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as server_socket:
            try:
                server_socket.bind(("127.0.0.1", port))
            except OSError:
                continue
            return port

    raise RuntimeError("没有找到可用端口，请关闭其他占用 8501-8599 的程序后重试。")


def open_browser_later(url: str) -> None:
    """稍等片刻后打开浏览器，给 Streamlit 服务一点启动时间。"""

    time.sleep(2.5)
    webbrowser.open(url)


def configure_streamlit(port: int) -> None:
    """配置 Streamlit 桌面运行参数，避免打包后误进入前端开发模式。"""

    # PyInstaller 打包后必须明确关闭开发模式，否则 Streamlit 可能只启动后端，
    # 导致浏览器访问首页时出现 404。
    os.environ["STREAMLIT_GLOBAL_DEVELOPMENT_MODE"] = "false"
    os.environ["STREAMLIT_BROWSER_GATHER_USAGE_STATS"] = "false"
    os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
    os.environ["STREAMLIT_SERVER_ADDRESS"] = "localhost"
    os.environ["STREAMLIT_SERVER_PORT"] = str(port)
    os.environ["STREAMLIT_SERVER_FILE_WATCHER_TYPE"] = "none"


def main() -> None:
    """检查资源文件，启动 Streamlit，并自动打开本地网页。"""

    app_path = find_resource(Path("app.py"))
    model_path = find_resource(MODEL_RELATIVE_PATH)

    if app_path is None:
        show_error("没有找到 app.py，安装包可能不完整。")
        return

    if model_path is None:
        show_error(
            "没有找到训练好的模型文件：models/garbage_resnet18.pth。\n"
            "请确认安装包是否完整，或重新安装软件。"
        )
        return

    port = find_available_port()
    url = f"http://localhost:{port}"

    configure_streamlit(port)

    if os.environ.get("GARBAGE_APP_NO_BROWSER") != "1":
        browser_thread = threading.Thread(
            target=open_browser_later,
            args=(url,),
            daemon=True,
        )
        browser_thread.start()

    from streamlit import config as streamlit_config
    from streamlit.web import bootstrap

    # 这里再直接覆盖一次配置。PyInstaller 打包后，Streamlit 会因为路径不在
    # site-packages 里而误判为开发模式；开发模式不会挂载首页静态资源。
    streamlit_config.set_option("global.developmentMode", False)
    streamlit_config.set_option("logger.level", "info")
    streamlit_config.set_option("server.fileWatcherType", "none")

    flag_options = {
        "global_developmentMode": False,
        "logger_level": "info",
        "server_port": port,
        "server_address": "localhost",
        "server_headless": True,
        "server_fileWatcherType": "none",
        "browser_gatherUsageStats": False,
    }
    bootstrap.run(str(app_path), False, [], flag_options)


if __name__ == "__main__":
    main()
