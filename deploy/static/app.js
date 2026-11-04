'use strict';
const $ = id => document.getElementById(id);
let choosingFolder = false, applyingTask = false, connected = false;
let status = null, inspected = null, folder = null, preview = null, previewKey = '', addressKey = '', initialized = false, noticeTimer;
const phases = {empty:'未加载',checking:'检查中',loading:'加载中',ready:'已就绪',unloading:'卸载中',error:'加载失败'};
const initialCameras = $('camera-grid').innerHTML;
const initialState = $('state-values').innerHTML;
let stateHistory = [];
const axisColors = ['#768b63','#c59d72','#9198ba'];
function text(id, value) { $(id).textContent = value; }
function notice(message) { text('folder-error',message);$('folder-error').hidden=!message;clearTimeout(noticeTimer); text('notice', message); $('notice').hidden = !message; if(message) noticeTimer = setTimeout(()=>{$('notice').hidden=true;},6000); }
function el(tag, content, cls) { const node=document.createElement(tag); if(content!==undefined)node.textContent=content;if(cls)node.className=cls;return node; }
async function api(path, body) {
  const response = await fetch('/api/'+path, body===undefined?{cache:'no-store'}:{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(body)});
  const data=await response.json();if(!response.ok)throw new Error(data.error||response.statusText);return data;
}
function action(id,fn){$(id).addEventListener('click',async()=>{try{notice('');await fn();}catch(error){notice(error.message);}});}
function checkpointInfo(info){
  const summary=`${info.policy_name==='vla'?'VLA':'Action Policy'} · ${info.chunk_size} steps · ${info.rtc_enabled?'RTC ≤ '+info.rtc_max_delay_steps:'无 RTC'}`;
  text('checkpoint-info',summary);$('checkpoint-info').title=summary;
  $('robot').replaceChildren(...info.robots.map(name=>{const option=el('option',name);option.value=name;return option;}));
  $('robot').value=info.robot_name;$('robot').disabled=false;
}
async function inspect(){inspected=await api('inspect',{path:$('checkpoint-path').value.trim()});$('checkpoint-path').value=inspected.path;checkpointInfo(inspected);return inspected;}
action('inspect',inspect);
action('load',async()=>{
  $('load').disabled=true;
  try{if(!inspected||inspected.path!==$('checkpoint-path').value.trim())await inspect();
    await api('load',{path:inspected.path,robot_name:$('robot').value,dtype:$('dtype').value,use_compile:$('compile').checked});await refreshStatus();
  }finally{if(status?.phase==='empty'||status?.phase==='error')$('load').disabled=false;}
});
action('unload',async()=>{await api('unload',{});await refreshStatus();});
action('apply-task',async()=>{
  if(applyingTask)return;
  applyingTask=true;renderTaskFeedback();
  try{await api('task',{task:$('task').value});await refreshStatus();}
  finally{applyingTask=false;renderTaskFeedback();}
});
function renderTaskFeedback(){
  const view=!connected?{label:'服务断开',tone:'error'}:applyingTask?{label:'正在应用…',tone:'pending'}:describeTaskStatus(status,$('task').value);
  text('task-state',view.label);$('task-state').dataset.tone=view.tone;
  $('apply-task').disabled=!connected||applyingTask||status?.metadata.policy_name==='action_policy'||!$('task').value.trim();
  $('task-details').hidden=status?.metadata.policy_name==='action_policy';
}
$('task').addEventListener('input',renderTaskFeedback);
$('task').addEventListener('keydown',event=>{if((event.ctrlKey||event.metaKey)&&event.key==='Enter')$('apply-task').click();});
async function openPathBrowser(){
  const roots=status?.roots||[];$('root').replaceChildren(...roots.map(path=>{const option=el('option',path);option.value=path;return option;}));
  const start=$('checkpoint-path').value.trim()||folder?.path||roots[0];
  try{await browse(start);}catch(error){await browse(roots[0]);notice(error.message);}
  $('folder-dialog').showModal();
}
action('browse-path',openPathBrowser);
action('browse',async()=>{
  if(choosingFolder)return;
  if(!status?.native_picker_available){await openPathBrowser();return;}
  choosingFolder=true;$('browse').disabled=true;
  try{
    const result=await api('choose-folder',{path:$('checkpoint-path').value.trim()});
    if(result.path){$('checkpoint-path').value=result.path;await inspect();}
  }catch(error){await openPathBrowser();notice(error.message);}
  finally{choosingFolder=false;$('browse').disabled=!['empty','error'].includes(status?.phase);}
});
async function browse(path){
  text('folder-error','');$('folder-error').hidden=true;
  folder=await api('browse?path='+encodeURIComponent(path));$('folder-path').value=folder.path;$('parent-folder').disabled=!folder.parent;
  $('folders').replaceChildren(...folder.directories.map(item=>{const button=el('button',undefined,'folder');button.append(el('span',item.name),el('small',item.checkpoint?'权重':'打开'));button.onclick=()=>browse(item.path).catch(error=>notice(error.message));return button;}));
  if(!folder.directories.length)$('folders').append(el('p','无子目录','muted'));
}
$('root').onchange=()=>browse($('root').value).catch(error=>notice(error.message));
action('go-folder',()=>browse($('folder-path').value.trim()));
$('folder-path').addEventListener('keydown',event=>{if(event.key==='Enter'){event.preventDefault();$('go-folder').click();}});
action('parent-folder',()=>browse(folder.parent));action('close-folder',()=>$('folder-dialog').close());
action('select-folder',async()=>{if($('folder-path').value.trim()!==folder.path)await browse($('folder-path').value.trim());$('checkpoint-path').value=folder.path;await inspect();$('folder-dialog').close();});
function showAddresses(addresses){
  const key=JSON.stringify(addresses);if(key===addressKey)return;addressKey=key;
  // Prefer a bound LAN address over loopback and virtual/benchmark interfaces.
  const lan=addresses.find(a=>a.reachable_bind&&/^(192\.168\.|10\.|172\.(1[6-9]|2\d|3[01])\.)/.test(a.ip));
  const primary=lan||addresses.find(a=>a.reachable_bind)||addresses[0];text('primary-ip',primary?.ip||'—');
  $('addresses').replaceChildren(...addresses.map(address=>{
    const row=el('div',undefined,'address');row.append(el('span',address.interface+(address.reachable_bind?'':' · 未绑定'),'address-name'),el('code',address.policy_url),el('small',address.page_url));
    const button=el('button','复制','secondary');button.title='复制 client 地址';
    button.onclick=async()=>{try{await navigator.clipboard.writeText(address.policy_url);button.textContent='已复制';}catch{notice(address.policy_url);}};
    row.append(button);return row;
  }));
}
async function refreshStatus(){
  const s=await api('status');status=s;connected=true;
  const locked=!['empty','error'].includes(s.phase),loading=['checking','loading','unloading'].includes(s.phase);
  text('phase',s.busy?'推理中':phases[s.phase]);$('phase').classList.toggle('loading',loading||s.busy);
  for(const id of ['load','inspect','browse','browse-path','checkpoint-path','dtype','compile'])$(id).disabled=locked||(id==='browse'&&choosingFolder);
  text('load',s.phase==='ready'?'模型已加载':s.phase==='checking'?'检查中…':s.phase==='loading'?'加载中…':s.phase==='unloading'?'卸载中…':'加载模型');
  $('load').classList.toggle('is-ready',s.phase==='ready');
  $('robot').disabled=locked||!inspected;$('unload').disabled=s.phase!=='ready'||!!s.client||s.busy;
  const actionOnly=s.metadata.policy_name==='action_policy';$('task').disabled=actionOnly;$('apply-task').disabled=actionOnly;
  text('policy-badge',s.metadata.policy_name==='action_policy'?'ACTION POLICY':s.metadata.policy_name==='vla'?'VLA':'LOCAL');
  text('task-hint',actionOnly?'Action Policy 不使用任务文本':'下一次推理生效 · 保留 RTC 前缀');
  $('apply-task').title='Ctrl / ⌘ + Enter';text('task-version',actionOnly?'VISION + STATE':'VLA');
  text('submitted',actionOnly?'—':s.task_version?'v'+s.task_version:'—');
  text('active-task',s.active_task_version!==null?'v'+s.active_task_version:s.busy?'AP':'—');
  text('result-task',s.result_task_version!==null?'v'+s.result_task_version:actionOnly&&s.infer_ms!==null?'AP':'—');
  text('saved-task',actionOnly?'':s.task||'');$('saved-task').title=s.task||'';
  text('latency',s.infer_ms===null?'—':s.infer_ms.toFixed(1)+' ms');
  text('prefix',s.metadata.prefix_conditioned?s.rtc_prefix_length+'/'+s.metadata.rtc_training_max_delay_steps:s.phase==='ready'?'未启用':'—');
  text('chunk',s.metadata.checkpoint_chunk_size||'—');
  text('client-status',s.client?'已连接':'未连接');$('client-status').title=s.client||'';
  text('model-hint',s.error||(loading?'请稍候':s.phase==='ready'?(s.metadata.policy_name==='vla'&&!s.task?'等待设置任务':s.client?'Client 已连接':'等待 client'):'选择本地 hf_ckpt'));$('model-hint').title=s.error||'';
  if(s.checkpoint&&locked){
    const path=s.checkpoint.path,parts=path.split('/').filter(Boolean);text('model-name',parts.slice(-2).join(' / '));$('model-name').title=path;
    text('checkpoint-info',`${actionOnly?'Action Policy':'VLA'} · ${s.checkpoint.robot_name}`);$('checkpoint-info').title=path;$('checkpoint-path').value=path;
    if($('robot').value!==s.checkpoint.robot_name){const option=el('option',s.checkpoint.robot_name);option.value=s.checkpoint.robot_name;$('robot').replaceChildren(option);}
    $('dtype').value=s.checkpoint.dtype;$('compile').checked=s.checkpoint.use_compile;
  }else{text('model-name','未选择权重');$('model-name').title='';}
  showAddresses(s.addresses);
  if(!initialized){if(s.task)$('task').value=s.task;initialized=true;}
  renderTaskFeedback();
}
function clearPreview(){
  if(!preview)return;preview=null;previewKey='';$('camera-grid').innerHTML=initialCameras;delete $('camera-grid').dataset.keys;delete $('camera-grid').dataset.count;
  stateHistory=[];$('state-values').innerHTML=initialState;text('sequence','—');text('observation-info','等待观测');
}
function cameraLabel(key){
  const short=key.replace('observation.images.','').replace(' / history','');
  const label=/head|top/.test(short)?'头部':/left/.test(short)?'左腕':/right/.test(short)?'右腕':short;
  return label+(key.endsWith('/ history')?' · 历史':'');
}
function renderCameras(p){
  const cameraOrder={'左腕':0,'头部':1,'右腕':2};
  const keys=Object.keys(p.images).filter(key=>!key.endsWith(' / history'))
    .sort((a,b)=>(cameraOrder[cameraLabel(a)]??3)-(cameraOrder[cameraLabel(b)]??3));
  const selected=keys.map(key=>$('head-frame').value==='history'&&p.images[key+' / history']?key+' / history':key);
  const grid=$('camera-grid');
  $('head-frame').disabled=!Object.keys(p.images).some(key=>key.endsWith(' / history'));
  if(!selected.length){grid.replaceChildren(el('div','无可用图像','xyz-empty'));delete grid.dataset.keys;delete grid.dataset.count;return;}
  if(grid.dataset.keys!==JSON.stringify(selected)){
    grid.replaceChildren(...selected.map(key=>{const card=el('div',undefined,'camera'),img=el('img');img.alt=key;card.append(img,el('span',cameraLabel(key),'camera-label'));card.title=key;return card;}));
    grid.dataset.keys=JSON.stringify(selected);grid.dataset.count=selected.length;
  }
  [...grid.children].forEach((card,index)=>{card.querySelector('img').src=p.images[selected[index]];});
}
$('head-frame').onchange=()=>{if(preview)renderCameras(preview);};
function drawSparkline(canvas, group){
  const width=canvas.getBoundingClientRect().width,height=52,dpr=window.devicePixelRatio||1;
  canvas.width=Math.max(1,width*dpr);canvas.height=height*dpr;
  const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);
  const all=stateHistory.flatMap(sample=>group.indices.map(index=>sample[group.key]?.[index])).filter(Number.isFinite);
  ctx.strokeStyle='#e6e9df';ctx.lineWidth=1;ctx.beginPath();ctx.moveTo(0,26);ctx.lineTo(width,26);ctx.stroke();
  if(!all.length)return;
  const low=Math.min(...all),high=Math.max(...all),range=high-low;
  group.indices.forEach((index,axis)=>{
    ctx.strokeStyle=axisColors[axis];ctx.lineWidth=1.4;ctx.beginPath();let connected=false;
    stateHistory.forEach((sample,i)=>{
      const value=sample[group.key]?.[index];if(!Number.isFinite(value)){connected=false;return;}
      const x=4+i/Math.max(stateHistory.length-1,1)*(width-8),y=range?46-(value-low)/range*40:26;
      if(connected)ctx.lineTo(x,y);else ctx.moveTo(x,y);connected=true;
    });ctx.stroke();
  });
  canvas.title=`XYZ · ${low.toFixed(3)} – ${high.toFixed(3)}`;
}
function renderXYZ(p){
  const groups=status?.schema.state_xyz||[];
  $('state-values').replaceChildren(...groups.map(group=>{
    const block=el('div',undefined,'state-block'),row=el('div',undefined,'xyz-row');block.title=group.key;
    block.append(el('h3',group.label));
    group.indices.forEach((index,axis)=>{const cell=el('div',undefined,'xyz-value'),value=p.states[group.key]?.[index];cell.append(el('span',['X','Y','Z'][axis]),el('strong',Number.isFinite(value)?value.toFixed(3):'—'));row.append(cell);});
    const canvas=el('canvas',undefined,'state-sparkline');canvas.setAttribute('aria-label',group.label+' XYZ 近期变化');block.append(row,canvas);return block;
  }));
  if(!groups.length)$('state-values').append(el('div','未配置 TCP XYZ','xyz-empty'));
  $('state-values').querySelectorAll('canvas').forEach((canvas,i)=>drawSparkline(canvas,groups[i]));
}
window.addEventListener('resize',()=>{if(preview)renderXYZ(preview);});
function renderPreview(p){
  if(!p){clearPreview();return;}
  const key=p.session_id+':'+p.observation_epoch+':'+p.sequence;if(key===previewKey)return;
  if(preview?.session_id!==p.session_id||preview?.observation_epoch!==p.observation_epoch)stateHistory=[];
  stateHistory.push(p.states);if(stateHistory.length>100)stateHistory.shift();
  preview=p;previewKey=key;text('sequence',p.sequence);
  const summary=`${new Date(p.received_at*1000).toLocaleTimeString()}${p.observation_tick!==undefined?' · tick '+p.observation_tick:''} · ${p.task_version===null?'Action Policy':'task v'+p.task_version}${p.errors.length?' · 图像异常':''}`;
  text('observation-info',summary);$('observation-info').title=p.errors.join('; ');
  renderCameras(p);renderXYZ(p);
}
async function poll(){
  try{await refreshStatus();renderPreview(await api('observation'));}catch(error){connected=false;text('phase','服务断开');text('client-status','服务断开');renderTaskFeedback();}
  if(preview){const age=Date.now()/1000-preview.received_at;text('freshness',(age>2?'已过期 · ':'')+age.toFixed(1)+' s');}else text('freshness','等待观测');
  setTimeout(poll,250);
}
poll();
