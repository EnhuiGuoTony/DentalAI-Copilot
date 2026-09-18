import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';
import { ChatConnectionResponse, DoxImportRequest, DoxImportSummary, DoxPreviewResponse, Patient, TimelineItem } from '../models/api.models';
import { UserProfile, Appointment, AppointmentInput, Conversation, ConversationState, AgentEvent } from '../models/api.models';
import { SessionState } from './session-state.service';

@Injectable({ providedIn: 'root' })
export class DentalAiApiService {
  private readonly baseUrl = 'http://localhost:8000/api';

  constructor(private readonly http: HttpClient) {}
  private readonly session = inject(SessionState);

  me(): Observable<UserProfile> { return this.http.get<UserProfile>(`${this.baseUrl}/auth/me`); }
  login(username: string, password: string): Observable<UserProfile> {
    return this.http.post<UserProfile>(`${this.baseUrl}/auth/login`, { username, password });
  }
  register(username: string, password: string, display_name: string): Observable<UserProfile> {
    return this.http.post<UserProfile>(`${this.baseUrl}/auth/register`, { username, password, display_name });
  }
  updateProfile(display_name: string): Observable<UserProfile> {
    return this.http.patch<UserProfile>(`${this.baseUrl}/auth/me`, { display_name });
  }
  password(current_password: string, new_password: string): Observable<void> {
    return this.http.post<void>(`${this.baseUrl}/auth/password`, { current_password, new_password });
  }
  logout(): Observable<void> { return this.http.post<void>(`${this.baseUrl}/auth/logout`, {}); }
  appointments(): Observable<Appointment[]> { return this.http.get<Appointment[]>(`${this.baseUrl}/appointments`); }
  saveAppointment(input: AppointmentInput, existing?: Appointment): Observable<unknown> {
    if (!existing) return this.http.post(`${this.baseUrl}/appointments`, input);
    const { patient_id, ...data } = input;
    return this.http.put(`${this.baseUrl}/appointments/${existing.id}`, {
      operation: 'update_appointment', appointment_id: existing.id, patient_id, data, expected_version: existing.version
    });
  }
  deleteAppointment(item: Appointment): Observable<unknown> {
    return this.http.delete(`${this.baseUrl}/appointments/${item.id}`, {
      params: { patient_id: item.patient_id, expected_version: item.version }
    });
  }
  conversations(): Observable<Conversation[]> { return this.http.get<Conversation[]>(`${this.baseUrl}/conversations`); }
  createConversation(patient_ids: string[]): Observable<Conversation> {
    return this.http.post<Conversation>(`${this.baseUrl}/conversations`, { patient_ids });
  }
  conversation(id: string): Observable<ConversationState> {
    return this.http.get<ConversationState>(`${this.baseUrl}/conversations/${id}`);
  }

  /** fetch 支持带 Cookie 的 POST 流；缓冲不完整帧，不能假设一个网络块就是一条事件。 */
  async stream(id: string, action: 'messages' | 'resume' | 'continue', body: object,
               onEvent: (event: AgentEvent) => void, signal: AbortSignal): Promise<void> {
    const response = await fetch(`${this.baseUrl}/conversations/${id}/${action}`, {
      method: 'POST', credentials: 'include', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body), signal
    });
    if (response.status === 401) this.session.user.set(null);
    if (!response.ok || !response.body) throw new Error(`请求失败 (${response.status})`);
    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';
    let terminal = false;
    try {
      while (true) {
        const { done, value } = await reader.read();
        buffer += decoder.decode(value, { stream: !done });
        let boundary: number;
        while ((boundary = buffer.indexOf('\n\n')) >= 0) {
          const frame = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);
          const data = frame.split('\n').filter(line => line.startsWith('data:')).map(line => line.slice(5).trim()).join('\n');
          if (data) {
            const event = JSON.parse(data) as AgentEvent;
            terminal ||= ['approval', 'result', 'error'].includes(event.type);
            onEvent(event);
          }
        }
        if (done) break;
      }
      if (!terminal) throw new Error('连接中断，请刷新会话检查执行状态。');
    } finally { reader.releaseLock(); }
  }

  patients(): Observable<Patient[]> {
    return this.http.get<Patient[]>(`${this.baseUrl}/patients`);
  }

  timeline(patientId: string): Observable<TimelineItem[]> {
    return this.http.get<TimelineItem[]>(`${this.baseUrl}/patients/${patientId}/timeline`);
  }

  connectChat(): Observable<ChatConnectionResponse> {
    return this.http.post<ChatConnectionResponse>(`${this.baseUrl}/chat/connect`, {});
  }

  previewDox(): Observable<DoxPreviewResponse> {
    return this.http.get<DoxPreviewResponse>(`${this.baseUrl}/dox-import/preview`);
  }

  importDox(request: DoxImportRequest): Observable<DoxImportSummary> {
    return this.http.post<DoxImportSummary>(`${this.baseUrl}/dox-import/run`, request);
  }
}
