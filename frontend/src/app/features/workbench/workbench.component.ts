import { CommonModule } from '@angular/common';
import { Component, OnInit, OnDestroy, signal, computed, effect, ElementRef, viewChild } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize, firstValueFrom } from 'rxjs';
import { DentalAiApiService } from '../../core/api/dentalai-api.service';
import { ChatMessage, DoxImportSummary, DoxPreviewResponse, Patient } from '../../core/models/api.models';
import { Appointment, AppointmentInput, Conversation, PendingReview, AgentEvent, AgentAnswer } from '../../core/models/api.models';

@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './workbench.component.html',
  styleUrl: './workbench.component.scss'
})
export class WorkbenchComponent implements OnInit, OnDestroy {
  // 页面切换只改变展示，保留正在运行的 SSE、会话和未提交表单。
  readonly section = signal<'agent' | 'appointments' | 'imports'>('agent');
  readonly patientSearch = signal('');
  readonly filteredPatients = computed(() => {
    const query = this.patientSearch().trim().toLowerCase();
    return this.patients().filter(patient =>
      [patient.name, patient.patient_number, patient.medical_record_number, patient.dox_patient_id]
        .some(value => value?.toLowerCase().includes(query)));
  });
  readonly reviewVisible = signal(false);
  private readonly approvalDialog = viewChild<ElementRef<HTMLDialogElement>>('approvalDialog');
  // showModal 提供焦点约束和背景隔离；业务审批状态仍以服务端 checkpoint 为准。
  private readonly syncApprovalDialog = effect(() => {
    const dialog = this.approvalDialog()?.nativeElement;
    if (!dialog) return;
    if (this.review() && this.reviewVisible()) {
      if (!dialog.open) dialog.showModal();
    } else if (dialog.open) dialog.close();
  });

  /** Escape 和稍后处理都只关闭展示，不向服务端发送审批决定。 */
  dismissReview(event?: Event): void {
    event?.preventDefault();
    if (!this.loading()) this.reviewVisible.set(false);
  }
  private readonly maxMessageChars = 4000;
  // 历史由服务端 checkpoint 持久化；浏览器仅保留当前展示内容。
  readonly messages = signal<ChatMessage[]>([]);
  readonly input = signal('');
  readonly connectionLabel = signal('连接模型');
  readonly connected = signal(false);
  readonly connecting = signal(false);
  readonly loading = signal(false);
  readonly patients = signal<Patient[]>([]);
  readonly patientsLoading = signal(false);
  readonly patientsError = signal('');
  readonly doxPreview = signal<DoxPreviewResponse | null>(null);
  readonly previewLoading = signal(false);
  readonly importLoading = signal(false);
  readonly importResult = signal<DoxImportSummary | null>(null);
  readonly importError = signal('');
  readonly importPatientLimit = signal(10);
  readonly selectedPatientIds = signal<string[]>([]);
  readonly appointments = signal<Appointment[]>([]);
  readonly conversations = signal<Conversation[]>([]);
  readonly activeConversation = signal<string | null>(null);
  readonly review = signal<PendingReview | null>(null);
  readonly events = signal<string[]>([]);
  readonly status = signal('就绪');
  readonly agentError = signal('');
  readonly canContinue = signal(false);
  readonly finalAnswer = signal<AgentAnswer | null>(null);
  readonly appointmentError = signal('');
  readonly appointmentBusy = signal(false);
  private streamController?: AbortController;
  decisions: ('approve' | 'reject')[] = [];
  appointmentPatient = ''; appointmentStart = ''; appointmentEnd = ''; appointmentReason = '';
  appointmentStatus: Appointment['status'] = 'scheduled';
  editingAppointment: Appointment | undefined;

  constructor(private readonly api: DentalAiApiService) {}

  ngOnInit(): void {
    this.loadPatients();
    this.loadAppointments();
    this.loadConversations();
  }

  ngOnDestroy(): void { this.streamController?.abort(); }

  loadAppointments(): void {
    this.api.appointments().subscribe({ next: items => this.appointments.set(items), error: () => this.appointmentError.set('无法读取预约。') });
  }
  patientName(id: string): string { return this.patients().find(p => p.id === id)?.name ?? id; }
  appointmentLabel(status: Appointment['status']): string {
    return { scheduled: '已预约', checked_in: '已到诊', completed: '已完成', cancelled: '已取消' }[status];
  }
  editAppointment(item: Appointment): void {
    this.editingAppointment = item; this.appointmentPatient = item.patient_id;
    // datetime-local 接受本地墙上时间；提交时再转换为明确的 UTC ISO 时间。
    const local = (value: string) => { const d = new Date(value); return new Date(d.getTime() - d.getTimezoneOffset() * 60000).toISOString().slice(0, 16); };
    this.appointmentStart = local(item.starts_at); this.appointmentEnd = local(item.ends_at);
    this.appointmentReason = item.reason; this.appointmentStatus = item.status;
  }
  resetAppointment(): void {
    this.editingAppointment = undefined; this.appointmentPatient = ''; this.appointmentStart = '';
    this.appointmentEnd = ''; this.appointmentReason = ''; this.appointmentStatus = 'scheduled';
  }
  saveAppointment(): void {
    if (this.appointmentBusy()) return;
    const start = new Date(this.appointmentStart), end = new Date(this.appointmentEnd);
    if (!this.appointmentPatient || !this.appointmentReason.trim() || !Number.isFinite(start.getTime()) || end <= start || !Number.isFinite(end.getTime())) {
      this.appointmentError.set('请选择患者、填写原因和有效起止时间。'); return;
    }
    const data: AppointmentInput = { patient_id: this.appointmentPatient, starts_at: start.toISOString(), ends_at: end.toISOString(), reason: this.appointmentReason, status: this.appointmentStatus };
    this.appointmentBusy.set(true); this.appointmentError.set('');
    this.api.saveAppointment(data, this.editingAppointment).pipe(finalize(() => this.appointmentBusy.set(false))).subscribe({
      next: () => { this.resetAppointment(); this.loadAppointments(); },
      error: e => this.appointmentError.set(e.status === 409 ? '时间冲突或记录已变化，请刷新后重试。' : '保存预约失败。')
    });
  }
  deleteAppointment(item: Appointment): void {
    if (this.appointmentBusy() || !window.confirm(`删除 ${this.patientName(item.patient_id)} 的这条预约？`)) return;
    this.appointmentBusy.set(true);
    this.api.deleteAppointment(item).pipe(finalize(() => this.appointmentBusy.set(false))).subscribe({
      next: () => this.loadAppointments(), error: () => this.appointmentError.set('删除失败，记录可能已变化，请刷新。')
    });
  }
  loadConversations(): void {
    this.api.conversations().subscribe({ next: items => this.conversations.set(items), error: () => this.agentError.set('无法读取历史会话。') });
  }
  openConversation(id: string): void {
    if (this.loading()) return;
    this.loading.set(true); this.agentError.set('');
    this.api.conversation(id).pipe(finalize(() => this.loading.set(false))).subscribe({
      next: state => {
        this.activeConversation.set(id); this.messages.set(state.messages); this.setReview(state.review);
        this.canContinue.set(state.can_continue); this.events.set([]); this.finalAnswer.set(null);
        this.selectedPatientIds.set(this.conversations().find(c => c.id === id)?.patient_ids ?? []);
        this.status.set(state.review ? '等待审批' : state.can_continue ? '执行中断，可继续' : '已恢复会话');
      }, error: () => this.agentError.set('无法恢复会话。')
    });
  }
  private setReview(review: PendingReview | null): void {
    this.review.set(review);
    this.reviewVisible.set(!!review);
    // 默认拒绝：每个动作必须由用户主动选择批准，不能批量默认同意。
    this.decisions = review?.actions.map(() => 'reject') ?? [];
  }
  private handleEvent(event: AgentEvent): void {
    if (event.type === 'status' || event.type === 'tool') {
      const text = `${event.tool_name ? event.tool_name + ' · ' : ''}${event.message}`;
      this.status.set(text); this.events.update(items => [...items.slice(-99), text]);
    } else if (event.type === 'approval') {
      this.setReview(event.review); this.status.set(event.message);
    } else if (event.type === 'result' && event.result) {
      this.finalAnswer.set(event.result); this.setReview(null); this.canContinue.set(false);
      this.messages.update(items => [...items, { role: 'assistant', content: event.result!.answer }]);
      this.status.set(event.mock ? '完成（Mock 模式）' : '完成');
    } else if (event.type === 'error') { this.agentError.set(event.message); this.status.set('执行失败，请刷新会话'); }
  }
  private async runStream(action: 'messages' | 'resume' | 'continue', body: object): Promise<void> {
    const id = this.activeConversation();
    if (!id) return;
    this.loading.set(true); this.agentError.set(''); this.streamController = new AbortController();
    try { await this.api.stream(id, action, body, e => this.handleEvent(e), this.streamController.signal); }
    catch (error) { if (!this.streamController.signal.aborted) this.agentError.set(error instanceof Error ? error.message : '连接中断，请刷新会话。'); }
    finally { this.loading.set(false); this.loadPatients(); this.loadAppointments(); }
  }
  async submitReview(): Promise<void> {
    const review = this.review();
    if (!review || this.loading()) return;
    await this.runStream('resume', { interrupt_id: review.interrupt_id, decisions: this.decisions.map(type => ({ type })) });
  }
  async continueRun(): Promise<void> { if (!this.loading()) await this.runStream('continue', {}); }

  loadPatients(): void {
    if (this.patientsLoading()) return;
    this.patientsLoading.set(true);
    this.patientsError.set('');
    this.api.patients()
      .pipe(finalize(() => this.patientsLoading.set(false)))
      .subscribe({
        next: patients => this.patients.set(patients),
        error: () => this.patientsError.set('无法读取患者信息，请确认后端已启动。')
      });
  }

  previewDox(): void {
    if (this.previewLoading() || this.importLoading()) return;
    this.previewLoading.set(true);
    this.importError.set('');
    this.api.previewDox()
      .pipe(finalize(() => this.previewLoading.set(false)))
      .subscribe({
        next: preview => this.doxPreview.set(preview),
        error: () => this.importError.set('无法连接 DOX 数据源，请检查 DOX_MYSQL_URL。')
      });
  }

  importDox(): void {
    if (this.importLoading()) return;
    const limit = Math.min(500, Math.max(1, Number(this.importPatientLimit()) || 1));
    this.importLoading.set(true);
    this.importError.set('');
    this.importResult.set(null);
    this.api.importDox({
      patient_limit: limit,
      include_patient_history: true,
      include_global_knowledge: true,
      deidentify: false
    })
      .pipe(finalize(() => this.importLoading.set(false)))
      .subscribe({
        next: result => {
          this.importResult.set(result);
          this.loadPatients();
          this.previewDox();
        },
        error: () => this.importError.set('导入失败，请查看后端日志或先执行预览。')
      });
  }

  age(patient: Patient): string {
    if (!patient.date_of_birth) return '—';
    const birth = new Date(`${patient.date_of_birth}T00:00:00`);
    if (Number.isNaN(birth.getTime())) return '—';
    const today = new Date();
    let age = today.getFullYear() - birth.getFullYear();
    const birthdayNotReached = today.getMonth() < birth.getMonth()
      || (today.getMonth() === birth.getMonth() && today.getDate() < birth.getDate());
    if (birthdayNotReached) age--;
    return age >= 0 ? `${age} 岁` : '—';
  }

  connect(): void {
    if (this.connecting() || this.connected()) return;

    this.connecting.set(true);
    this.connectionLabel.set('连接中…');
    this.api.connectChat()
      .pipe(finalize(() => this.connecting.set(false)))
      .subscribe({
        next: response => {
          this.connected.set(response.connected);
          this.connectionLabel.set(response.connected ? (response.mock ? 'Mock 模型已连接' : '模型已连接') : '连接不可用 · 重试');
        },
        error: () => this.connectionLabel.set('连接失败 · 重试')
      });
  }

  async send(): Promise<void> {
    const message = this.input().trim();
    if (!message || this.loading() || !this.connected() || this.review() || this.canContinue()) return;
    const userTurn: ChatMessage = {
      role: 'user',
      content: message.slice(0, this.maxMessageChars)
    };
    this.messages.set([...this.messages(), userTurn]);
    this.input.set('');
    this.loading.set(true);

    try {
      if (!this.activeConversation()) {
        const conversation = await firstValueFrom(this.api.createConversation(this.selectedPatientIds()));
        this.activeConversation.set(conversation.id); this.loadConversations();
      }
      await this.runStream('messages', { message: userTurn.content });
    } catch { this.agentError.set('创建会话失败。'); }
    finally { this.loading.set(false); }
  }

  togglePatient(patientId: string, checked: boolean): void {
    if (this.activeConversation()) return;
    this.selectedPatientIds.update(ids => checked ? [...new Set([...ids, patientId])] : ids.filter(id => id !== patientId));
  }

  summarizeSelected(): void {
    if (!this.selectedPatientIds().length) return;
    this.input.set('请总结所选患者的病历、治疗史、诊断和最近笔记。');
    this.send();
  }

  clear(): void {
    if (this.loading()) return;
    this.messages.set([]);
    this.activeConversation.set(null); this.setReview(null); this.events.set([]); this.canContinue.set(false);
    this.finalAnswer.set(null); this.agentError.set(''); this.status.set('新会话：请选择患者范围');
  }
}
