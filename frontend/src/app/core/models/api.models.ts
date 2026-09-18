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
  tokenUsage?: TokenUsage;
}

export interface TokenUsage {
  input_tokens: number;
  output_tokens: number;
  total_tokens: number;
}

export interface ChatResponse {
  reply: string;
  provider: string;
  mock: boolean;
  tool_trace: string[];
  token_usage: TokenUsage;
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
/** 登录、预约与 Agent 协议和后端 Pydantic 模型保持一致。 */
export interface UserProfile { id: string; username: string; display_name: string; created_at: string; }
export interface Appointment {
  id: string; patient_id: string; starts_at: string; ends_at: string; reason: string;
  status: 'scheduled' | 'checked_in' | 'completed' | 'cancelled'; version: string; created_at: string;
}
export type AppointmentInput = Pick<Appointment, 'patient_id' | 'starts_at' | 'ends_at' | 'reason' | 'status'>;
export interface Conversation { id: string; patient_ids: string[]; created_at: string; }
export interface ReviewAction { name: string; arguments: Record<string, unknown>; description: string; }
export interface PendingReview { interrupt_id: string; actions: ReviewAction[]; }
export interface AgentAnswer { answer: string; evidence_ids: string[]; limitations: string[]; }
export interface AgentEvent {
  type: 'status' | 'tool' | 'approval' | 'result' | 'error'; message: string;
  tool_name: string | null; review: PendingReview | null; result: AgentAnswer | null; mock: boolean;
}
export interface ConversationState { id: string; messages: ChatMessage[]; review: PendingReview | null; can_continue: boolean; }
