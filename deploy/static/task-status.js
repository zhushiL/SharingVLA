'use strict';

// Describe the draft separately from the task version actually used by inference.
function describeTaskStatus(state, draft) {
  if (!state) return {label:'连接中', tone:'idle'};
  if (state.metadata.policy_name === 'action_policy') return {label:'Action Policy · 无需任务', tone:'idle'};
  const value = draft.trim(), saved = (state.task || '').trim();
  if (value !== saved) return {label:value ? '待应用' : '请输入新任务', tone:'pending'};
  if (!saved) return {label:'输入任务以开始', tone:'idle'};
  if (state.phase === 'error') return {label:'任务已保存 · 模型加载失败', tone:'error'};
  if (state.phase !== 'ready') return {label:'任务已保存 · 等待模型就绪', tone:'pending'};
  if (state.busy && state.active_task_version === state.task_version) return {label:'新任务正在推理', tone:'active'};
  if (state.error) return {label:'推理异常 · 查看顶部提示', tone:'error'};
  if (state.result_task_version === state.task_version) return {label:'新任务已用于推理', tone:'success'};
  if (!state.client) return {label:'任务已保存 · 等待 Client', tone:'idle'};
  return {label:'等待下一轮推理', tone:'pending'};
}
