import { CommonModule } from '@angular/common';
import { Component, OnInit, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';
import { DentalAiApiService } from '../../core/api/dentalai-api.service';
import { ChatMessage, DoxImportSummary, DoxPreviewResponse, Patient } from '../../core/models/api.models';

@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './workbench.component.html',
  styleUrl: './workbench.component.scss'
})
export class WorkbenchComponent implements OnInit {
  private readonly maxHistoryMessages = 4;
  private readonly maxMessageChars = 4000;
  private conversationHistory: ChatMessage[] = [];

  // The visible transcript keeps every turn; only this smaller array is sent back.
  readonly messages = signal<ChatMessage[]>([]);
  readonly input = signal('Explain this project in one minute for an AI Engineer interview.');
  readonly connectionLabel = signal('Connect model');
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

  constructor(private readonly api: DentalAiApiService) {}

  ngOnInit(): void {
    this.loadPatients();
  }

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
    this.connectionLabel.set('Connecting...');
    this.api.connectChat()
      .pipe(finalize(() => this.connecting.set(false)))
      .subscribe({
        next: response => {
          this.connected.set(response.connected);
          this.connectionLabel.set(response.connected ? 'Model connected' : 'Connection unavailable');
        },
        error: () => this.connectionLabel.set('Connection unavailable')
      });
  }

  send(): void {
    const message = this.input().trim();
    if (!message || this.loading() || !this.connected()) return;

    const history = this.conversationHistory
      .slice(-this.maxHistoryMessages)
      .map(({ role, content }) => ({ role, content }));
    const userTurn: ChatMessage = {
      role: 'user',
      content: message.slice(0, this.maxMessageChars)
    };
    this.messages.set([...this.messages(), userTurn]);
    this.input.set('');
    this.loading.set(true);

    this.api.chat(userTurn.content, history, this.selectedPatientIds())
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: response => {
          const assistantTurn: ChatMessage = {
            role: 'assistant',
            content: response.reply,
            tokenUsage: response.token_usage
          };
          this.conversationHistory = [...this.conversationHistory, userTurn, assistantTurn]
            .slice(-this.maxHistoryMessages);
          this.messages.set([...this.messages(), assistantTurn]);
        },
        error: () => {
          this.connected.set(false);
          this.connectionLabel.set('Request failed - reconnect');
        }
      });
  }

  togglePatient(patientId: string, checked: boolean): void {
    this.selectedPatientIds.update(ids => checked ? [...new Set([...ids, patientId])] : ids.filter(id => id !== patientId));
  }

  summarizeSelected(): void {
    if (!this.selectedPatientIds().length) return;
    this.input.set('请总结所选患者的病历、治疗史、诊断和最近笔记。');
    this.send();
  }

  clear(): void {
    this.messages.set([]);
    this.conversationHistory = [];
  }
}
