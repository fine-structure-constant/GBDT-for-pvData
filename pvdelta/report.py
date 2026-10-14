"""Static scientific plots and self-contained human-readable audit reports."""
import base64
import html
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def create_report(data,audit,issues,results,predictions,importance,summary,config,out):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180})
    navy='#254B6E'; gold='#C18531'; gray='#8B949E'; red='#B44C43'
    fig,axs=plt.subplots(2,2,figsize=(13,9),layout='constrained')
    fields=['cas_number','molecular_formula','smiles','molecular_weight','additive_abbreviation']
    missing=data[fields].isna().sum().sort_values()
    axs[0,0].barh(missing.index,missing.values,color=navy)
    axs[0,0].set(xlabel='Missing records',title=f'Missing source fields (n={len(data):,})')
    for i,v in enumerate(missing): axs[0,0].text(v+10,i,str(v),va='center')
    status=data.composition_parse_status.value_counts()
    axs[0,1].barh(status.index[::-1],status.values[::-1],color=navy)
    axs[0,1].set(xlabel='Records',title='Closed-grammar host parsing')
    axs[1,0].hist(data.pce_delta_abs,bins=np.arange(-20,46,1),alpha=.5,color=gray,label='All source records')
    axs[1,0].hist(data.loc[data.strict_eligible,'pce_delta_abs'],bins=np.arange(-20,46,1),alpha=.7,color=navy,label='Strict baseline cohort')
    axs[1,0].axvline(0,color='black',lw=.8)
    axs[1,0].set(xlabel='Absolute PCE change (percentage points)',ylabel='Records',title='Target distribution; negative results retained')
    axs[1,0].legend(frameon=False)
    top=issues.head(9).sort_values('affected_records')
    axs[1,1].barh(top.issue_code,top.affected_records,color=gold)
    axs[1,1].tick_params(axis='y',labelsize=8)
    axs[1,1].set(xlabel='Affected records (issues overlap)',title='Most frequent audit findings')
    fig.savefig(out/'data_quality.png',bbox_inches='tight');fig.savefig(out/'data_quality.svg',bbox_inches='tight');plt.close(fig)
    fig,axs=plt.subplots(2,2,figsize=(12,9),layout='constrained')
    actual=predictions.pce_delta_abs.to_numpy()
    predicted=predictions.prediction_GBDT.to_numpy()
    hb=axs[0,0].hexbin(actual,predicted,gridsize=36,mincnt=1,cmap='Blues',bins='log')
    lim=[min(actual.min(),predicted.min())-1,max(actual.max(),predicted.max())+1]
    axs[0,0].plot(lim,lim,'--',color=gray,lw=1)
    axs[0,0].set(xlim=lim,ylim=lim,xlabel='Observed delta (pp)',ylabel='Out-of-fold predicted delta (pp)',title=f'Grouped nested CV, n={len(actual):,}')
    plt.colorbar(hb,ax=axs[0,0],label='Records (log scale)')
    m=summary['primary_metrics']
    axs[0,0].text(.03,.97,f'MAE = {m["MAE_pp"]:.3f} pp\nRMSE = {m["RMSE_pp"]:.3f} pp\nR² = {m["R2"]:.3f}',transform=axs[0,0].transAxes,va='top',bbox={'facecolor':'white','edgecolor':'none','alpha':.85})
    axs[0,1].scatter(predicted,actual-predicted,s=8,alpha=.25,c=navy,rasterized=True)
    axs[0,1].axhline(0,color=gray,lw=1)
    axs[0,1].set(xlabel='Out-of-fold predicted delta (pp)',ylabel='Observed minus predicted (pp)',title='Residuals reveal remaining variation')
    names={'GBDT':'GBDT + control PCE','Dummy_mean':'Mean baseline','Dummy_median':'Median baseline','Control_only_GBDT':'Control PCE only','Materials_only_GBDT':'Materials only'}
    r=results[results.validation.eq('material_pair_group')].sort_values('MAE_pp',ascending=False)
    axs[1,0].barh([names[x] for x in r.model],r.MAE_pp,color=[navy if x=='GBDT' else gray for x in r.model])
    axs[1,0].set(xlabel='Out-of-fold MAE (pp; lower is better)',title='Same outer groups, baseline comparison')
    for i,v in enumerate(r.MAE_pp): axs[1,0].text(v+.02,i,f'{v:.3f}',va='center')
    axs[1,0].set_xlim(0,r.MAE_pp.max()*1.2)
    imp=importance.sort_values('mean')
    axs[1,1].barh(imp.feature_group,imp['mean'],xerr=imp['std'],color=gold,capsize=3)
    axs[1,1].axvline(0,color=gray,lw=.8)
    axs[1,1].set(xlabel='MAE increase after joint permutation (pp)',title='Held-out feature-group importance (mean ± SD)')
    fig.savefig(out/'model_performance.png',bbox_inches='tight');fig.savefig(out/'model_performance.svg',bbox_inches='tight');plt.close(fig)
    p=json.loads((out/'provenance.json').read_text(encoding='utf-8'))
    ci=summary['primary_group_bootstrap_95pct']
    issue_lines='\n'.join(f'| {r.issue_code} | {int(r.affected_records)} | {r.action} |' for _,r in issues.iterrows())
    metric_lines='\n'.join(f'| {r.validation} | {r.model} | {r.MAE_pp:.4f} | {r.RMSE_pp:.4f} | {r.R2:.4f} |' for _,r in results.iterrows())
    examples=audit[audit.issue_code.isin(['ABBREVIATION_FORMULA_MISMATCH','PCE_ABOVE_REVIEW_CEILING','HOST_STOICHIOMETRY_OR_FAMILY'])].head(12)
    example_lines='\n'.join(f'| {r.record_id} | {int(r.source_excel_row)} | {r.issue_code} | {r.raw_value} | {r.detail} |' for _,r in examples.iterrows())
    report=f'''# 钙钛矿添加剂数据形式化与 GBDT 基线

本轮目标为 `pce_delta_abs = pce_after - pce_before`，单位为**效率百分点 pp**。用户确认预测时已知 `pce_before`。这是文献记录的关联预测，尚不能解释为添加剂的因果效应。

## 数据范围与溯源

- 当前文件只有 **{p['rows']:,} 条记录、{p['columns']} 个源字段**，工作表 `{p['sheet']}`。没有 DOI 或 paper_id，无法确认论文数，也无法完成论文级独立验证。
- 原文件不改写；每条记录带 `record_id`、原 Excel 行号。源 SHA-256：`{p['sha256']}`。
- 源数据按目标值递减排序，因此不得按文件前后顺序拆训练/测试集。
- 全量规范化保留 {len(data):,} 条；宽松筛查候选 {int(data.screened_eligible.sum()):,} 条；保守训练集 **{summary['n']:,} 条**。筛查候选不是已验证无误的数据。
- 保守训练集仍包含化学元数据缺失和名称多CID等未解决问题。严格只表示满足本版规则，不表示文献事实均已复核。

## 基于材料原理的形式化规则

1. **器件先分层。** 常规 n-i-p 与 p-i-n 作为结构类别；叠层、组件、DSSC/PEC 和无法明确分类者保留但不并入本轮常规单结基线。仅靠名称不能恢复 ETL/HTL、面积或测量条件。
2. **主体组成按 A/B/X 名义位点拆分。** `CH3NH3`→`MA`，`HC(NH2)2`/`CH(NH2)2`→`FA`，采用封闭词典；Cs/FA/MA 等进入 A，Pb/Sn 等进入 B，I/Br/Cl/SCN 等进入 X。掺杂金属按组成记账，并不代表已证明占据 B 位。
3. **系数与比例分开保存。** 例如 `Cs0.05_FA0.8_MA0.15_Pb_I2.55_Br0.45` 的 A 比例为 0.05/0.8/0.15，X 比例为 0.85/0.15；另保留 A/B/X 系数和与 X/B。绝不只留归一化比例而丢失错误计量。
4. **未知不等于零。** `FA_MA_Pb_I3-x_Clx` 只保留已知物种，A 与 X 比例未知；不假设 FA:MA=1:1，不设 x=0。完全未解析或多相的行不输出伪确定比例。`I2_Br` 可按常规化学式解释为 I2Br1。
5. **区分结构族。** 检查 ABX3、RP 候选、DJ 候选及 Cs2AgBiX6 类双钙钛矿。RP 检查大阳离子2、总A=B+1、X=3B+1；DJ采用词典中明确二价间隔阳离子的候选规则。候选计量不等于晶体结构实测结论。容差 {config['stoichiometry_tolerance']} 是可配置的数据审查规则。
6. **添加剂保留分子/盐整体，同时拆显式离子片段。** 如 `C[NH3+].[I-]` 得到阳离子与阴离子；`CN.I` 仅记成两个中性表示片段，不反推真实电离状态。没有溶液环境信息时不能推断实际物种。分子式给出元素计数、质量和形式电荷。SMILES 只进行词法片段/原子清点，未做 RDKit 化学图验证，不计算伪造的氢键受体数、指纹或logP。
7. **混合添加剂不复制标签。** `DMSO; MACl` 保留为一次实验；不能拆成两条共享同一 ΔPCE 的独立样本。源表只有一个化学元数据对象时，屏蔽混合配方的整体分子描述符。剂量与各组分比例仍缺失。
8. **校验化学匹配。** 对明确常见缩写采用透明的小词典检查分子式；核对分子量、CAS校验位、SMILES非氢原子。发现冲突不静默改成预计值，保存原始证据并隔离。没有标记不代表身份一定正确。
9. **不虚构物性。** 本轮不从名义配方强行推算带隙、容忍因子、形成能或配位能力；缺少所需半径、结构和相信息。

## 已发现的问题

下表按受影响记录计数；同一行可有多个问题，不能把计数相加当作剔除行数。

| 问题代码 | 记录数 | 本轮处理 |
|---|---:|---|
{issue_lines}

具体证据（行号对应原工作表）：

| ID | Excel行 | 问题 | 原值 | 说明 |
|---|---:|---|---|---|
{example_lines}

化学元数据存在明显错配：例如 MACl 被映射为 CH4N2S（硫脲），KI 被映射为 H2O。分子式、SMILES和CID彼此自洽，也可能整体对应了错误添加剂。因此不能仅凭SMILES可解析就宣布匹配正确。

35% 是本轮常规单结记录的**审查阈值**，不是该材料的精确理论极限，也不是现有效率纪录。缺少光谱、光强及器件信息时，超阈值只隔离待核查。接近零的对照效率也不自动乘100；保守训练暂时隔离单位歧义。所有负 ΔPCE 保留，不按提升与否筛选。

同一已知材料组合的不同结果可能来自剂量、工艺、实验室或测量差异；未把它们当成错误删除，未平均标签。完全重复及规范化后的相同观察值仅在训练集中保留一次，无法证明它们是否属于不同论文。

## 模型、验证与信息泄漏控制

- 使用 scikit-learn `GradientBoostingRegressor`，即 GBDT。输入为 {summary['n_raw_features']} 个原始特征列；类别编码后的维数由各训练折决定。
- 采用**外层5折材料组合分组 + 内层3折选参**。同一“结构+规范化主体组成+添加剂实验名称”不跨训练和验证折。化学映射有错误，所以验证分组不直接信任CID。名称别名未穷尽合并，仍可能低估关联泄漏。
- 缺失数值在各训练折拟合中位数并添加缺失指示；类别编码仅在训练折拟合，稀有类别合并。GBDT无需标准化。没有在全数据上先填补、编码或筛选特征。
- 明确排除 `pce_after`、相对提升率、目标重算值、原行号、ID/CAS/CID数值等。`pce_before` 已获用户确认可用，但它和目标有数学耦合，不能把其重要性解读成材料机理。
- 同折比较均值/中位数基线、仅对照效率GBDT、去掉对照效率的材料GBDT。消融模型沿用主模型该折选出的参数，并非各自独立最优调参。
- 另做“未见添加剂名称”分组测试和随机行拆分诊断，使用预先固定候选0，不能将随机拆分成绩当作新论文泛化能力。
- 模型重要性来自验证折**整组特征共同置换**，避免只置换某一个比例破坏同位点组分关系。它表达预测依赖，不是因果贡献；误差条为折/置换间SD。

| 验证 | 模型 | MAE (pp) | RMSE (pp) | R² |
|---|---|---:|---:|---:|
{metric_lines}

主模型 OOF：MAE **{m['MAE_pp']:.3f} pp**，RMSE **{m['RMSE_pp']:.3f} pp**，R² **{m['R2']:.3f}**。
按材料组合对固定OOF预测进行1000次bootstrap的95%区间：MAE [{ci['MAE_pp'][0]:.3f}, {ci['MAE_pp'][1]:.3f}]；R² [{ci['R2'][0]:.3f}, {ci['R2'][1]:.3f}]。区间没有重新训练整个嵌套流程，不代表完整的模型选择不确定性。

最终全量拟合参数：`{json.dumps(summary['final_parameters'])}`。保存模型用于复现；性能来自OOF而非该模型对训练数据的拟合。

![数据质量](data_quality.png)

![模型表现](model_performance.png)

## 重整后的数据

- `cleaned_records.csv`：全量原字段、规范字段、位点系数/比例、审计标记和纳入规则。
- `host_components_long.csv`：一行一个主体名义物种，保留记录ID、位点、系数、比例与解析状态。
- `additive_fragments_long.csv`：一行一个原SMILES显式片段，记录形式电荷及阳/阴/中性角色；不表示真实溶液物种已确认。
- `training_features.csv`：保守训练集和明确输入列，含目标与分组元数据，建模时以 `feature_dictionary.csv`/保存模型的特征名单为准。
- `issues.csv`：逐条问题、原行号、证据、处理建议；`composition_dictionary.csv` 用于人工补词典。
- `oof_predictions.csv`：逐条交叉验证预测、残差、折号。不能将其当成全新外部测试集。

## 下一轮优先回补

1. **论文 DOI、原表/图位置、对照-处理配对ID**：先确认前后PCE来自同条件可比器件，允许同论文内多条实验。
2. **修复添加剂实体匹配**：审查明确缩写冲突及一名多CID；记录盐、反离子、水合物、纯度与匹配证据。
3. **剂量与加入位置**：数值、单位、mol%分母、所加层/溶液、复合添加剂比例。只有名称无法确定是晶体组分、结晶助剂、界面钝化剂还是传输层掺杂剂。
4. **主体名义/实测组成及工艺**：分开存储前驱体投料与最终薄膜组成，补溶剂、反溶剂、退火、ETL/HTL、膜厚等。
5. **测量上下文**：面积、光强与光谱、扫描方向/速率、稳态MPP、冠军值或均值、样本数与误差、年份、实验室。

在上述上下文补齐之前，本轮结果可用来检查数据流程、定位缺陷和建立基线，不能直接作为推荐实验配方的证据。

## 复现与参考

```powershell
# 系统Python提供sklearn/matplotlib；Excel读取使用pandas/openpyxl，只读源文件。
python -m unittest discover -s tests -v
python run_project.py --prepare-only
python run_project.py
```

版本：`{json.dumps(summary['versions'])}`。配置集中在 `config.json`，改变规则后需要重新清洗、训练和导出表格。

- [钙钛矿数据库的数据粒度、隐藏变量与FAIR原则，Nature Energy](https://www.nature.com/articles/s41560-021-00941-3)
- [GBDT官方实现](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html)
- [GroupKFold官方文档](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GroupKFold.html)
- [预处理与信息泄漏官方说明](https://scikit-learn.org/stable/common_pitfalls.html)
'''
    (out/'方法与数据问题报告.md').write_text(report,encoding='utf-8')
    issue_table=issues[['issue_code','affected_records','action']].to_html(index=False,escape=True,border=0)
    metric_table=results[['validation','model','MAE_pp','RMSE_pp','R2']].to_html(index=False,float_format=lambda v:f'{v:.4f}',border=0)
    imgs=[]
    for filename in ['data_quality.png','model_performance.png']:
        imgs.append('<img alt="'+filename+'" src="data:image/png;base64,'+base64.b64encode((out/filename).read_bytes()).decode()+'">')
    body=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>钙钛矿 ΔPCE 数据审计与 GBDT</title><style>body{{font:16px/1.7 system-ui,"Microsoft YaHei",sans-serif;max-width:1200px;margin:40px auto;padding:0 24px;color:#233340}}h1{{font-size:30px}}h2{{margin-top:36px}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:9px 12px;border-bottom:1px solid #dbe0e4;text-align:left}}th{{background:#edf1f5}}img{{max-width:100%;height:auto}}.note{{padding:16px 20px;background:#fff4df}}code{{font-family:monospace}}a{{color:#254b6e}}</style><h1>钙钛矿 ΔPCE 数据审计与 GBDT</h1><p>4,272条源记录；保守基线 {summary['n']:,} 条。目标为绝对效率增量，单位百分点。预测时已知对照效率。</p><p class="note">缺少论文DOI、剂量、添加位置及测量条件。存在化学身份错配。本轮是探索性关联模型；材料组合分组不等于论文独立验证。</p><h2>基于材料原理的预处理</h2><p>按名义A/B/X位点拆分物种，分别保存原始系数、位内比例与计量检查；未知配比保留空值。显式SMILES片段保留阳离子、阴离子和中性角色。混合添加剂不复制实验标签，错误映射不静默修正。</p><h2>数据问题</h2>{issue_table}{imgs[0]}<h2>验证结果</h2><p>主模型采用5折外层材料组合分组和3折内层选参；缺失填补与类别编码在训练折内完成。排除 pce_after 及相对提升率等泄漏字段。</p>{metric_table}{imgs[1]}<p>重要性是预测依赖，并非因果贡献。随机拆分仅作诊断；未见添加剂测试按名称分组。</p><h2>后续数据修复</h2><p>优先补充 DOI与配对ID，修复名称/CID/分子式错配，补剂量及分母、加入位置、ETL/HTL、溶剂与退火、光照与稳态测量协议。完整方法、逐行审计和复现步骤见同目录的 Markdown 报告。</p></html>'''
    (out/'结果总览.html').write_text(body,encoding='utf-8')
    # One intermediate JSON is used by the artifact-tool workbook builder.
    review_columns=['record_id','source_excel_row','strict_eligible','screened_eligible','exclusion_reasons','issue_codes',
        'solar_cell_structure','architecture','perovskite_composition','composition_normalized','composition_parse_status','composition_topology',
        'A_species','A_coefficients_json','A_sum','A_ratio_known','B_species','B_coefficients_json','B_sum','B_ratio_known',
        'X_species','X_coefficients_json','X_sum','X_ratio_known','A_to_B','X_to_B','nominal_charge_residual_per_B',
        'additive_abbreviation','additive_mixture_flag','cas_number','pubchem_id','smiles','molecular_formula','molecular_weight',
        'additive_formula_status','additive_mw_g_mol','additive_fragment_count','additive_cation_fragments','additive_anion_fragments',
        'pce_before','pce_after','pce_delta_abs','pce_delta_relative_pct','delta_recomputed_pp','delta_residual_pp',
        'exact_duplicate','canonical_duplicate','material_pair_group','additive_group']
    tables={
        'Summary':[['Metric','Value'],['Source records',len(data)],['Strict baseline records',summary['n']],['Target unit','percentage points'],['Known control PCE',True],['Grouped OOF MAE (pp)',m['MAE_pp']],['Grouped OOF RMSE (pp)',m['RMSE_pp']],['Grouped OOF R2',m['R2']],['Paper count','unknown; no DOI'],['Source SHA256',p['sha256']],['Important limitation','No DOI, dose, location or measurement context']],
        'Cleaned records':json.loads(data[review_columns].to_json(orient='split',index=False)),
        'Host components':json.loads(pd.read_csv(out/'host_components_long.csv').to_json(orient='split',index=False)),
        'Additive fragments':json.loads(pd.read_csv(out/'additive_fragments_long.csv').to_json(orient='split',index=False)),
        'Issues':json.loads(audit.to_json(orient='split',index=False)),
        'Issue summary':json.loads(issues.to_json(orient='split',index=False)),
        'Features':json.loads(pd.read_csv(out/'feature_dictionary.csv').to_json(orient='split',index=False)),
        'OOF predictions':json.loads(predictions.to_json(orient='split',index=False)),
        'Model metrics':json.loads(results.to_json(orient='split',index=False))
    }
    (out/'workbook_payload.json').write_text(json.dumps(tables,ensure_ascii=False,allow_nan=False),encoding='utf-8')
    print('Reports and figures saved.',flush=True)
