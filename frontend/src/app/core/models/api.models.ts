export interface Patient {
  id: string;
  name: string;
  date_of_birth: string | null;
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

export interface XrayImage {
  id: string;
  patient_id: string;
  case_id: string;
  file_path: string;
  width: number | null;
  height: number | null;
  created_at: string;
}

export interface XrayFinding {
  id: string;
  image_id: string;
  case_id: string;
  category: string;
  confidence: number;
  tooth_number: string | null;
  bbox: { x: number; y: number; w: number; h: number };
  polygon: Array<[number, number]> | null;
  review_status: string;
  created_at: string;
}

export interface EvidenceItem {
  source_type: string;
  source_id: string;
  snippet: string;
  score: number;
  metadata: Record<string, unknown>;
}

export interface AgentRunResponse {
  id: string;
  patient_id: string;
  case_id: string;
  question: string;
  evidence: EvidenceItem[];
  tool_trace: Array<{ tool: string; input: Record<string, unknown>; output_summary: string }>;
  output: {
    clinical_summary: string;
    suspected_findings: Array<{
      category: string;
      confidence: number;
      tooth_number: string | null;
      evidence_refs: string[];
    }>;
    relevant_history: string[];
    recommended_next_steps: string[];
    patient_friendly_explanation: string;
    limitations: string[];
  };
  created_at: string;
}

export interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
}

export interface ChatResponse {
  reply: string;
  provider: string;
  mock: boolean;
}
