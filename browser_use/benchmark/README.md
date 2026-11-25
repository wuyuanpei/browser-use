# Browser Use 基线测试台

测试台复用 Jev Ultrafast 的页面、110 条基线、批次和归档逻辑。页面以本机 Jev 8810 版本为参考，其前端对应源树 `8306149942e03c2ca9c97de7f645f57d74b79fc5`。用例分组取该页面的目录快照。原文件许可见 `LICENSE`。

从 Browser Use 仓库根目录启动。

```powershell
uv venv --python 3.12
uv pip install -r pyproject.toml --extra benchmark
.venv\Scripts\python.exe -m browser_use.benchmark --port 8824
```

打开 http://127.0.0.1:8824/#baseline。自由任务和基线测试均由 Browser Use 执行。浏览器默认使用独立临时配置。批次保存在当前目录的 `artifacts/baselines`，可用 `--artifacts` 指定目录。

默认使用调试配置。Browser Use 从原来的 `PLANNER_*` 读取执行模型，独立终验从 `TEXT_MODEL_*` 读取模型。最终报告沿用原来的模型、服务和专用凭据，仅显式选择时用于当次运行。可用 `--profile-env` 读取已有的本机 Jev 配置，也可以在仓库的 `.env` 配置相同变量。若真实用例需要登录，可用 `BROWSER_USE_BENCHMARK_USER_DATA_DIR` 指定专用 Chrome 配置目录。

## 复用正在运行的 Chrome

在 Chrome 打开 `chrome://inspect/#remote-debugging` 并开启远程调试。`DevToolsActivePort` 位于 Chrome 用户数据目录，可在 `chrome://version` 查看 `Profile Path` 并取其上一级目录。启动测试台时用 `--chrome-active-port-file` 传入该文件，或配置 `BROWSER_USE_BENCHMARK_CHROME_ACTIVE_PORT_FILE`。

```powershell
.venv\Scripts\python.exe -m browser_use.benchmark --port 8824 --chrome-active-port-file 'C:\path\to\Chrome User Data\DevToolsActivePort'
```

另开一个 PowerShell 窗口查询连接页。

```powershell
$origin = 'http://127.0.0.1:8824'
(Invoke-RestMethod "$origin/api/state").chrome_attachment.anchor_url
```

在要复用登录态的 Chrome 资料中打开上一步返回的链接。页面加载完成后发起连接，并在 Chrome 弹窗中批准。返回 `connected = True` 后可从测试台运行自由任务和基线。

```powershell
$html = (Invoke-WebRequest -UseBasicParsing "$origin/").Content
$token = [regex]::Match($html, '<meta name="demo-token" content="([^"]+)"').Groups[1].Value
$headers = @{ 'X-Demo-Token' = $token }
(Invoke-RestMethod "$origin/api/chrome/connect" -Method Post -Headers $headers -ContentType 'application/json' -Body '{}').chrome_attachment
```

每个用例创建自己的测试标签，连续用例共用已批准的连接。批次会记录 `browser_mode=attached_chrome`。批次结束后可以用相同的 POST 请求调用 `/api/chrome/disconnect`。连接断开后重新从 `/api/state` 获取链接，再按相同步骤连接。

## 运行记录

`engine.py` 管理 Browser Use 会话和运行事件。`tasks.py` 准备用例并从浏览器采集验收证据，`baseline.py` 保存批次、截图和 trace，`evaluation.py` 根据冻结的证据独立终验。历史归档里的 Planner、重规划和资料库由原版界面从 trace 回放。

可运行 `python -m browser_use.benchmark.reevaluate ATTEMPT_FOLDER` 复评历史归档。加上 `--prepare-only --output CONTEXT.json` 可以单独导出评估输入。

公开用例 79、83 和 91 使用测试邮箱或测试收货信息。实际跑这三条用例时，可以将用例数据替换为专用测试账号的值。
