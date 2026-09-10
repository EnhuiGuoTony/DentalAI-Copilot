import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { ChatConnectionResponse, ChatMessage, ChatResponse, ClinicalCase, DoxImportRequest, DoxImportSummary, DoxPreviewResponse, Patient, TimelineItem } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class DentalAiApiService {
  private readonly baseUrl = 'http://localhost:8000/api';

  constructor(private readonly http: HttpClient) {}

  patients(): Observable<Patient[]> {
    return this.http.get<Patient[]>(`${this.baseUrl}/patients`);
  }

  timeline(patientId: string): Observable<TimelineItem[]> {
    return this.http.get<TimelineItem[]>(`${this.baseUrl}/patients/${patientId}/timeline`);
  }

  chat(message: string, history: ChatMessage[], patientIds: string[] = []): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, { message, history, patient_ids: patientIds });
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
