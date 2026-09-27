import { Injectable, inject, signal } from '@angular/core';
import { ApiService } from './api.service';
import { ChapterSummary, JobDto, JobEventDto, JobStatus, JobUsage } from '../models/api.models';

export interface NowRunning {
  chapter: { index: string; total: string; title: string } | null;
  phase: string;
}

const FINISHED: JobStatus[] = ['succeeded', 'failed', 'cancelled'];

/** Watches one build end-to-end: opens the WebSocket, falls back to polling
 *  if it drops, and keeps the log/chapters/usage/now-running signals the
 *  progress panel renders. One instance for the app (providedIn: 'root') —
 *  only one build runs at a time server-side, so that's the natural scope. */
@Injectable({ providedIn: 'root' })
export class BuildStreamService {
  private readonly api = inject(ApiService);

  readonly jobId = signal<string | null>(null);
  readonly status = signal<JobStatus | 'idle'>('idle');
  readonly events = signal<JobEventDto[]>([]);
  readonly chapters = signal<ChapterSummary[]>([]);
  readonly usage = signal<JobUsage | null>(null);
  readonly nowRunning = signal<NowRunning | null>(null);
  readonly finishedJob = signal<JobDto | null>(null);
  /** Known as soon as intake completes — well before the build finishes —
   *  which is what lets a mid-build chapter card open its verdict dialog. */
  readonly channelSlug = signal<string | null>(null);

  private socket: WebSocket | null = null;
  private pollTimer: ReturnType<typeof setInterval> | null = null;
  private usageTimer: ReturnType<typeof setInterval> | null = null;
  private seenSeq = 0;

  /** On boot: pick up a build already running on the server, so a page
   *  reload during a long run doesn't look idle. */
  async reattachIfRunning(): Promise<void> {
    try {
      const payload = await this.api.get<{ builds: JobDto[] }>('/api/builds');
      const active = payload.builds.find((job) => job.status === 'running');
      if (active) this.start(active.id);
    } catch {
      /* server unreachable; health already reports it */
    }
  }

  start(jobId: string): void {
    this.stop();
    this.jobId.set(jobId);
    this.seenSeq = 0;
    this.events.set([]);
    this.chapters.set([]);
    this.usage.set(null);
    this.nowRunning.set(null);
    this.finishedJob.set(null);
    this.channelSlug.set(null);
    this.status.set('running');

    this.refreshUsage();
    this.usageTimer = setInterval(() => this.refreshUsage(), 4000);

    const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
    try {
      const socket = new WebSocket(`${protocol}://${location.host}/api/builds/${jobId}/stream`);
      this.socket = socket;
      socket.onmessage = (message) => {
        const payload = JSON.parse(message.data);
        if (payload.type === 'event') this.handleEvent(payload as JobEventDto);
        else if (payload.type === 'status') this.finish(payload as JobDto);
      };
      // A closed socket mid-run means the connection dropped, not that the
      // build ended; fall back to polling rather than freezing the UI.
      socket.onclose = () => { if (this.jobId() === jobId) this.startPolling(jobId); };
      socket.onerror = () => this.startPolling(jobId);
    } catch {
      this.startPolling(jobId);
    }
  }

  async cancel(): Promise<void> {
    const jobId = this.jobId();
    if (!jobId) return;
    await this.api.post(`/api/builds/${jobId}/cancel`);
  }

  stop(): void {
    if (this.socket) { try { this.socket.close(); } catch { /* already closed */ } this.socket = null; }
    if (this.pollTimer) { clearInterval(this.pollTimer); this.pollTimer = null; }
    if (this.usageTimer) { clearInterval(this.usageTimer); this.usageTimer = null; }
  }

  /** Show a workspace's chapters straight from the manifest, with no live
   *  job behind them — used by the library's "Chapters" action. */
  showChapters(slug: string, chapters: ChapterSummary[]): void {
    this.jobId.set(null);
    this.channelSlug.set(slug);
    this.chapters.set(chapters);
  }

  private startPolling(jobId: string): void {
    if (this.pollTimer) return;
    this.pollTimer = setInterval(async () => {
      try {
        const payload = await this.api.get<{ status: JobStatus; events: JobEventDto[] }>(
          `/api/builds/${jobId}/events?after=${this.seenSeq}`,
        );
        payload.events.forEach((event) => this.handleEvent(event));
        if (FINISHED.includes(payload.status)) {
          const job = await this.api.get<JobDto>(`/api/builds/${jobId}`);
          this.finish(job);
        }
      } catch {
        /* transient; the next tick retries */
      }
    }, 2000);
  }

  private handleEvent(event: JobEventDto): void {
    if (event.seq <= this.seenSeq) return;
    this.seenSeq = event.seq;
    this.events.update((events) => [...events, event]);
    if (event.kind === 'stage') this.updateNowRunning(event.message);
    if (event.kind === 'done' || event.kind === 'error') void this.refreshChapters();
    else if (event.kind === 'stage' && /Finished|Starting/.test(event.message)) void this.refreshChapters();
  }

  private updateNowRunning(message: string): void {
    const text = message.trim();
    const startMatch = text.match(/^\[(\d+)\/(\d+)\]\s+Starting:\s+(.+)\.\s+Elapsed\s+\S+,\s+ETA\b/);
    const phaseMatch = text.match(/^\[(\d+)\]\s+(.+)$/);
    const current = this.nowRunning();
    if (startMatch) {
      this.nowRunning.set({
        chapter: { index: startMatch[1], total: startMatch[2], title: startMatch[3] },
        phase: current?.phase ?? '',
      });
    } else if (phaseMatch) {
      this.nowRunning.set({
        chapter: current?.chapter ?? null,
        phase: phaseMatch[2].replace(/\s*\.\.\.$/, ''),
      });
    } else {
      // A top-level stage (intake, compiling, ...) rather than a per-chapter
      // one — it supersedes whatever chapter/phase was showing before.
      this.nowRunning.set({ chapter: null, phase: text });
    }
  }

  private async refreshChapters(): Promise<void> {
    const jobId = this.jobId();
    if (!jobId) return;
    let job: JobDto;
    try { job = await this.api.get<JobDto>(`/api/builds/${jobId}`); } catch { return; }
    if (job.channel_slug) this.channelSlug.set(job.channel_slug);

    let chapters = (job.chapters ?? []) as unknown as ChapterSummary[];
    // While the build is running the job has no outcomes yet, but the
    // workspace manifest is written per stage — read live progress from there.
    if (!chapters.length && job.channel_slug) {
      try {
        const payload = await this.api.get<{ chapters: ChapterSummary[] }>(
          `/api/workspaces/${job.channel_slug}/chapters`,
        );
        chapters = payload.chapters;
      } catch {
        /* workspace not created yet */
      }
    }
    this.chapters.set(chapters);
  }

  private async refreshUsage(): Promise<void> {
    const jobId = this.jobId();
    if (!jobId) return;
    try {
      const job = await this.api.get<JobDto>(`/api/builds/${jobId}`);
      this.usage.set(job.usage);
    } catch {
      /* transient */
    }
  }

  private async finish(job: JobDto): Promise<void> {
    this.stop();
    this.status.set(job.status);
    this.nowRunning.set(null);
    await this.refreshChapters();
    this.usage.set(job.usage);
    this.finishedJob.set(job);
  }
}
