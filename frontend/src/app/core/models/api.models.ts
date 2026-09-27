/** Types mirroring the JSON shapes bookforge.api.app / bookforge.api.jobs return.
 *  Kept as one file: these are wire contracts, not domain logic — they should
 *  change exactly when the FastAPI response shapes change, together. */

export interface HealthResponse {
  version: string;
  ready: boolean;
  warnings: string[];
  tools: { pdflatex: boolean; ffmpeg: boolean };
  themes: string[];
  settings: {
    theme: string;
    workspace_root: string;
    compile_latex: boolean;
    qa_max_iterations: number;
    video_extensions: string[];
    transcript_extensions: string[];
    youtube_cookies_configured: boolean;
  };
  models: Record<'analyst' | 'writer' | 'critic' | 'caption' | 'reflection', string>;
  concurrency: { chapters_parallel: number; media_prefetch_workers: number };
}

export interface PreviewItem {
  video_id: string;
  title: string;
  source: string;
  duration_sec: number | null;
  path: string | null;
  url: string | null;
}

export interface PreviewResponse {
  title: string;
  source_line: string;
  warnings: string[];
  count: number;
  items: PreviewItem[];
}

export interface UploadResult {
  path: string;
  saved: { name: string; bytes: number }[];
  rejected: string[];
  total_bytes: number;
}

export interface BuildRequest {
  source: string[];
  title?: string;
  theme?: string;
  max_videos?: number;
  compile_latex?: boolean;
  qa_iterations?: number;
}

export interface UsageRecord {
  calls: number;
  prompt_tokens: number;
  completion_tokens: number;
  failures: number;
}

export interface JobUsage {
  by_role: Record<string, UsageRecord>;
  total: UsageRecord;
}

export type JobStatus = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

export interface BookResult {
  title?: string;
  chapters_included: number;
  verified_chapters: number;
  mean_quality_score: number | null;
  overfull_boxes: number | null;
  pdf?: boolean;
}

export interface JobDto {
  id: string;
  source: string[];
  title: string;
  theme: string;
  status: JobStatus;
  created_at: string;
  started_at: string;
  finished_at: string;
  error: string;
  workspace_root: string;
  channel_slug: string;
  result: BookResult | null;
  chapters: unknown[];
  last_seq: number;
  usage: JobUsage;
}

export interface JobEventDto {
  seq: number;
  kind: 'stage' | 'info' | 'warning' | 'error' | 'done';
  message: string;
  at: string;
  data: Record<string, unknown>;
}

export interface ChapterDefect {
  severity: 'critical' | 'major' | 'minor';
  message: string;
  fix?: string;
}

export interface ChapterHistoryEntry {
  iteration: number;
  score: number;
  compiles: boolean;
  critical_defects: number;
  total_defects: number;
  fingerprint: string;
  note: string;
}

export interface ChapterVerdict {
  score: number;
  accepted: boolean;
  iterations: number;
  compiles: boolean;
  defects: ChapterDefect[];
  history: ChapterHistoryEntry[];
}

export interface ChapterSummary {
  video_id: string;
  chapter_number: number;
  title: string;
  status: string;
  source: string;
  qa_score: number | null;
  grounding_score: number | null;
  iterations: number | null;
  compiled: boolean;
  error: string | null;
  verdict: ChapterVerdict | null;
}

export interface ChaptersResponse {
  slug: string;
  title: string;
  chapters: ChapterSummary[];
}

export interface WorkspaceSummary {
  slug: string;
  title: string;
  source: string;
  created_at: string;
  chapters: number;
  verified: number;
  has_pdf: boolean;
}

export interface WorkspacesResponse {
  workspaces: WorkspaceSummary[];
}

/* ------------------------------------------------------------------ config */

export type ConfigFieldType = 'text' | 'password' | 'number' | 'select' | 'checkbox';

export interface ConfigField {
  key: string;
  group: string;
  type: ConfigFieldType;
  label: string;
  hint?: string;
  min?: number;
  max?: number;
  step?: number;
  options?: (string | number)[];
  secret?: boolean;
  is_set?: boolean;
  value: unknown;
}

export interface ConfigGroup {
  key: string;
  title: string;
}

export interface ConfigResponse {
  fields: ConfigField[];
  groups: ConfigGroup[];
  provider_env: Record<string, boolean>;
  env_file: string | null;
}

export interface ConfigUpdateResult {
  applied: string[];
  env_updated: string[];
}

export interface ConfigTestResult {
  ok: boolean;
  model: string;
  latency_ms: number;
  reply?: string;
  error?: string;
}
