export interface Patient {
  id: string;
  name: string;
  date_of_birth: string | null;
  dox_patient_id: string | null;
  patient_number: string | null;
  medical_record_number: string | null;
  address: string | null;
  created_at: string;
}

export interface ClinicalCase {
  id: string;
  patient_id: string;
  title: string;
  status: string;
  created_at: string;
}

export interface TimelineItem {
  type: string;
  id: string;
  title: string;
  content: string;
  created_at: string;
}

export interface ChatMessage {
  role: 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  reply: string;
  provider: string;
  mock: boolean;
  tool_trace: string[];
}

export interface ChatConnectionResponse {
  connected: boolean;
  provider: string;
  mock: boolean;
  detail: string;
}

export interface DoxPreviewResponse {
  connected: boolean;
  tables: Record<string, number | null>;
  warnings: string[];
}

export interface DoxImportRequest {
  patient_limit?: number;
  include_global_knowledge: boolean;
  include_patient_history: boolean;
  deidentify: boolean;
}

export interface DoxImportSummary {
  run_id: string | null;
  status: string;
  patients_requested: number;
  patients_imported: number;
  notes_imported: number;
  knowledge_chunks_imported: number;
  clinical_facts_imported: number;
  warnings: string[];
}
