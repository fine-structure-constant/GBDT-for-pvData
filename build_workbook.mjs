import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile, FileBlob} from '@oai/artifact-tool';

const config=JSON.parse(await fs.readFile('config.json','utf8'));
const out=path.resolve(config.output_dir);
const payload=JSON.parse(await fs.readFile(path.join(out,'workbook_payload.json'),'utf8'));
const repair=process.argv.includes('--repair-layout');
const wb=repair?await SpreadsheetFile.importXlsx(await FileBlob.load(path.join(out,'重整数据与审计.xlsx'))):Workbook.create();
const sheets=[];
const order=['Summary','Model metrics','Cleaned records','Host components','Additive fragments','Issues','Issue summary','Features','OOF predictions'];
for(const name of order){
  const source=payload[name];
  const matrix=Array.isArray(source)?source:[source.columns,...source.data];
  const s=repair?wb.worksheets.getItem(name):wb.worksheets.add(name);
  sheets.push({s,matrix});
  if(repair) continue;
  s.showGridLines=false;
  const range=s.getRangeByIndexes(0,0,matrix.length,matrix[0].length);
  range.values=matrix.map(row=>row.map(v=>typeof v==='string'&&v.startsWith('=')?"'"+v:v));
  range.format.font={name:'Arial',size:10,color:'#233340'};
  range.format.verticalAlignment='center';
  range.format.rowHeight=18;
  range.format.columnWidth=21;
  const header=s.getRangeByIndexes(0,0,1,matrix[0].length);
  header.format={fill:'#254B6E',font:{name:'Arial',size:10,bold:true,color:'#FFFFFF'},wrapText:true,rowHeight:52,verticalAlignment:'center',horizontalAlignment:'center'};
  if(name!=='Summary'){
    s.freezePanes.freezeRows(1);
    s.freezePanes.freezeColumns(Math.min(2,matrix[0].length));
    s.tables.add(range,true,'T_'+name.replaceAll(' ','_'));
    header.format.fill='#254B6E';
  }
  for(let c=0;c<matrix[0].length;c++){
    const h=String(matrix[0][c]);
    const col=s.getRangeByIndexes(0,c,matrix.length,1);
    if(/smiles|coefficients|exclusion|issue_codes|detail|action|raw_value|meaning|parameters/.test(h))col.format.columnWidth=42;
    if(/source_excel_row|fragment_index|fold/.test(h))col.format.columnWidth=15;
    const sample=matrix.slice(1).map(r=>r[c]).find(v=>v!==null);
    if(typeof sample==='number'){
      s.getRangeByIndexes(1,c,matrix.length-1,1).setNumberFormat(/count|row|fold|n_unique|n_train|n_test/.test(h)?'0':'0.0000');
      s.getRangeByIndexes(1,c,matrix.length-1,1).format.horizontalAlignment='right';
    }
  }
  if(name==='Summary'){
    s.tabColor='#254B6E';
    s.getRange('A1:A12').format.columnWidth=33;
    s.getRange('B1:B12').format.columnWidth=91;
    s.getRange('A1:B1').format.rowHeight=26;
    s.getRange('B6:B8').setNumberFormat('0.0000');
    const img=await fs.readFile(path.join(out,'model_performance.png'));
    s.images.add({dataUrl:'data:image/png;base64,'+img.toString('base64'),anchor:{from:{row:14,col:0},extent:{widthPx:1050,heightPx:790}}});
  }
  console.log('Prepared',name,matrix.length-1,'rows',matrix[0].length,'cols');
}
// Audit labels and explanations must be readable without opening each cell.
for(const name of ['Issues','Issue summary']){
  const entry=sheets.find(v=>v.s.name===name);
  const s=entry.s;
  const last=entry.matrix.length;
  const codeCol=name==='Issues'?'C':'A';
  s.getRange(`${codeCol}1:${codeCol}${last}`).format.columnWidth=53;
  if(name==='Issues'){
    s.getRange(`F2:H${last}`).format.wrapText=true;
    s.getRange(`G1:H${last}`).format.columnWidth=70;
    s.getRange(`G2:H${last}`).format.rowHeight=54;
  }else{
    s.getRange(`C1:C${last}`).format.columnWidth=70;
    s.getRange(`C2:C${last}`).format.wrapText=true;
    s.getRange(`C2:C${last}`).format.rowHeight=40;
    s.getRange(`D2:D${last}`).setNumberFormat('0');
    s.getRange(`E2:E${last}`).setNumberFormat('0.0%');
  }
}
const clean=sheets.find(v=>v.s.name==='Cleaned records');
clean.s.getRange(`F1:F${clean.matrix.length}`).format.columnWidth=78;
clean.s.getRange(`E2:F${clean.matrix.length}`).format.wrapText=true;
clean.s.getRange(`E2:F${clean.matrix.length}`).format.rowHeight=54;
wb.worksheets.getItem('Summary').getRange('B2:B3').setNumberFormat('0');
wb.worksheets.getItem('Model metrics').getRange('C2:C10').setNumberFormat('0');
const frag=sheets.find(v=>v.s.name==='Additive fragments');
frag.s.getRange(`C2:C${frag.matrix.length}`).setNumberFormat('0');
frag.s.getRange(`E2:E${frag.matrix.length}`).setNumberFormat('0');
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Summary!A1:B12',tableMaxRows:12,tableMaxCols:2,maxChars:4000})).ndjson);
console.log((await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:10},maxChars:1000})).ndjson);
await fs.mkdir(path.join(out,'workbook_previews'),{recursive:true});
for(const {s,matrix} of sheets){
  const preview=await wb.render({sheetName:s.name,range:s.name==='Summary'?'A1:B12':'A1:F5',scale:1.3,format:'png'});
  await fs.writeFile(path.join(out,'workbook_previews',s.name.replaceAll(' ','_')+'.png'),new Uint8Array(await preview.arrayBuffer()));
  console.log('Rendered',s.name);
}
const file=await SpreadsheetFile.exportXlsx(wb);
await file.save(path.join(out,'重整数据与审计.xlsx'));
console.log('Exported workbook');
