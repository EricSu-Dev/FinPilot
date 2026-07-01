"""本地开发与部署工具的根入口。

某些运行器要求仓库根目录存在一个顶层 `.py` 文件。本模块重新导出
`app.main` 中定义的 FastAPI 应用，使 `uvicorn main:app` 与
`uvicorn app.main:app` 行为一致。
"""
# 把 app/main.py 里面的变量 app 导入过来。
# 某些 Chroma 部署环境需要较新的 sqlite，可安装 pysqlite3 后自动替换；
# 本地未安装时继续使用标准库 sqlite3，避免 `python main.py` 直接启动失败。
try:
    __import__("pysqlite3")
except ModuleNotFoundError:
    pass
else:
    import sys

    sys.modules["sqlite3"] = sys.modules.pop("pysqlite3")

from app.main import app

#直接运行 python test.py, __name__ == "__main__"
#被导入 import test ,__name__ == 模块名"test"
#只有当前文件被 .py 直接运行的时候，才执行下面的代码
if __name__ == "__main__":
    """以项目标准的本地端口启动 API。"""
    #懒加载 ,只有真正启动服务器的时候才导入 uvicorn
    import uvicorn
    #host="0.0.0.0" 允许所有机器访问
    #reload=True 热更新,代码改了自动重启
    uvicorn.run("app.main:app", host="0.0.0.0", port=8094, reload=False)
