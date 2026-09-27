import { Injectable, computed, inject, signal } from '@angular/core';
import { ApiService } from './api.service';
import { BuildRequest, JobDto, PreviewResponse, UploadResult } from '../models/api.models';

export type SourceTab = 'upload' | 'folder' | 'youtube';

interface StagedFile {
  file: File;
}

/** Everything on the "before a build starts" side of the page: which source
 *  tab is active, staged/uploaded files, the folder path or YouTube URL, book
 *  options, and the last preview. A build's live progress is a separate
 *  concern — see BuildStreamService. */
@Injectable({ providedIn: 'root' })
export class BuildFormService {
  private readonly api = inject(ApiService);

  readonly activeTab = signal<SourceTab>('upload');
  readonly stagedFiles = signal<StagedFile[]>([]);
  readonly uploadedPath = signal<string | null>(null);
  readonly folderPath = signal('');
  readonly youtubeUrl = signal('');

  readonly bookTitle = signal('');
  readonly theme = signal('textbook');
  readonly maxVideos = signal<number | null>(null);
  readonly qaIterations = signal<number | null>(null);
  readonly compileLatex = signal(true);

  readonly preview = signal<PreviewResponse | null>(null);
  readonly previewError = signal<string | null>(null);
  readonly previewing = signal(false);
  readonly uploading = signal(false);

  readonly totalStagedBytes = computed(() =>
    this.stagedFiles().reduce((sum, f) => sum + f.file.size, 0));

  readonly canBuild = computed(() => (this.preview()?.count ?? 0) > 0);

  setTab(tab: SourceTab): void {
    this.activeTab.set(tab);
    this.clearPreview();
  }

  /** The source list POST /api/preview and POST /api/builds both expect, or
   *  null when the active tab has nothing usable yet. */
  currentSource(): string[] | null {
    switch (this.activeTab()) {
      case 'upload': {
        const path = this.uploadedPath();
        return path ? [path] : null;
      }
      case 'folder': {
        const value = this.folderPath().trim();
        return value ? [value] : null;
      }
      case 'youtube': {
        const value = this.youtubeUrl().trim();
        return value ? [value] : null;
      }
    }
  }

  addFiles(fileList: FileList | File[]): void {
    const existing = this.stagedFiles();
    const next = [...existing];
    for (const file of Array.from(fileList)) {
      if (!next.some((f) => f.file.name === file.name && f.file.size === file.size)) {
        next.push({ file });
      }
    }
    this.stagedFiles.set(next);
    this.uploadedPath.set(null);
    this.clearPreview();
  }

  removeFile(index: number): void {
    const next = [...this.stagedFiles()];
    next.splice(index, 1);
    this.stagedFiles.set(next);
    this.uploadedPath.set(null);
    this.clearPreview();
  }

  async upload(): Promise<UploadResult> {
    this.uploading.set(true);
    try {
      const form = new FormData();
      for (const { file } of this.stagedFiles()) form.append('files', file);
      const result = await this.api.postForm<UploadResult>('/api/upload', form);
      this.uploadedPath.set(result.path);
      return result;
    } finally {
      this.uploading.set(false);
    }
  }

  clearPreview(): void {
    this.preview.set(null);
    this.previewError.set(null);
  }

  async runPreview(): Promise<void> {
    const source = this.currentSource();
    if (!source) {
      throw new Error(
        this.activeTab() === 'upload'
          ? 'Upload the files to the server first.'
          : 'Enter a path or URL.',
      );
    }
    this.previewing.set(true);
    this.previewError.set(null);
    try {
      const preview = await this.api.post<PreviewResponse>('/api/preview', {
        source,
        title: this.bookTitle().trim(),
      });
      this.preview.set(preview);
    } catch (err) {
      this.preview.set(null);
      this.previewError.set(err instanceof Error ? err.message : String(err));
      throw err;
    } finally {
      this.previewing.set(false);
    }
  }

  async submitBuild(): Promise<JobDto> {
    const source = this.currentSource();
    if (!source) throw new Error('No source selected.');

    const body: BuildRequest = {
      source,
      title: this.bookTitle().trim(),
      theme: this.theme(),
      compile_latex: this.compileLatex(),
    };
    const maxVideos = this.maxVideos();
    if (maxVideos !== null && Number.isFinite(maxVideos)) body.max_videos = maxVideos;
    const qaIterations = this.qaIterations();
    if (qaIterations !== null && Number.isFinite(qaIterations)) body.qa_iterations = qaIterations;

    return this.api.post<JobDto>('/api/builds', body);
  }
}
