# 钙钛矿添加剂 ΔPCE 项目

本轮交付位于 `outputs/pvdelta_v1/`。先打开 `结果总览.html` 或 `方法与数据问题报告.md`，再按 `issues.csv` 的原 Excel 行号回查。

目标为绝对增量（百分点），预测时允许使用已知对照效率。当前源文件4272条记录，不含论文标识，不能确认对应论文数。最终数据集2722条属于本版保守规则筛出的探索性基线，仍有未解决的数据问题。

## 运行

Python依赖见 `requirements.txt`；本机已有的 `D:\.ENV\Python\python.exe` 可完成模型和图形任务。读取源Excel不写回源文件。

```powershell
python run_project.py --prepare-only
python run_project.py
python -m unittest discover -s tests -v
```

所有规则与参数集中在 `config.json`。代码按组成解析、数据审计、建模、报告分模块。不要只修改导出的CSV后重新训练；优先修复源数据副本或解析规则，保留变更证据，再重新运行全流程。

## 表格导出

`build_workbook.mjs` 使用 @oai/artifact-tool 从报告流程生成的 `workbook_payload.json` 导出一个审阅工作簿。需要已配置该库的Node环境；当前工作区的 `node_modules` 是对Codex内置依赖的目录连接，不包含项目自建依赖。

```powershell
& 'C:\Users\71861\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe' --max-old-space-size=8192 build_workbook.mjs
```

Excel提供审阅字段视图，完整宽表与完整训练特征保存为CSV。每次变更后都要重新导出Excel，避免旧版本混用。源文件SHA-256保存于 `provenance.json`。

## 结果边界

主性能来自5折外层材料组合分组、内层3折选参的OOF预测。缺失填补和编码均在训练折完成。材料组合分组无法替代论文级分组。未见添加剂测试按名称分组，名称别名未穷尽。最终全量拟合模型不另计训练集性能。

缺少剂量、加入位置、工艺与测量条件；化学信息错配尚未全部修复。本轮模型用于数据诊断与基线建立，不把重要性解释成因果机理，也不直接据此推荐配方。
