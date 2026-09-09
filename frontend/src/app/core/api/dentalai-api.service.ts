import { HttpClient } from '@angular/common/http';
import { Injectable } from '@angular/core';
import { Observable } from 'rxjs';
import { AgentRunResponse, ChatConnectionResponse, ChatMessage, ChatResponse, ClinicalCase, Patient, TimelineItem, XrayFinding, XrayImage } from '../models/api.models';

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

  createCase(patientId: string): Observable<ClinicalCase> {
    return this.http.post<ClinicalCase>(`${this.baseUrl}/patients/${patientId}/cases`, { title: 'AI X-ray review' });
  }

  uploadXray(caseId: string, file: File): Observable<XrayImage> {
    const data = new FormData();
    data.append('file', file);
    return this.http.post<XrayImage>(`${this.baseUrl}/cases/${caseId}/xray`, data);
  }

  analyze(caseId: string, imageId: string): Observable<XrayFinding[]> {
    return this.http.post<XrayFinding[]>(`${this.baseUrl}/cases/${caseId}/xray/${imageId}/analyze`, {});
  }

  runAgent(caseId: string, question: string): Observable<AgentRunResponse> {
    return this.http.post<AgentRunResponse>(`${this.baseUrl}/cases/${caseId}/agent/run`, { question });
  }

  imageUrl(imageId: string): string {
    return `${this.baseUrl}/images/${imageId}/file`;
  }

  chat(message: string, history: ChatMessage[]): Observable<ChatResponse> {
    return this.http.post<ChatResponse>(`${this.baseUrl}/chat`, { message, history });
  }

  connectChat(): Observable<ChatConnectionResponse> {
    return this.http.post<ChatConnectionResponse>(`${this.baseUrl}/chat/connect`, {});
  }
}
