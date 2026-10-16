import { Component, ElementRef, inject, signal, viewChild } from '@angular/core';
import { BuildFormService, SourceTab } from '../../../core/services/build-form.service';
import { fmtBytes, fmtDuration } from '../../../core/util/format';
import { HealthService } from '../../../core/services/health.service';

@Component({
  selector: 'app-source-panel',
  templateUrl: './source-panel.html',
})
export class SourcePanel {
  protected readonly form = inject(BuildFormService);
  protected readonly health = inject(HealthService);

  protected readonly dragOver = signal(false);
  protected readonly uploadError = signal<string | null>(null);
  private readonly fileInput = viewChild.required<ElementRef<HTMLInputElement>>('fileInput');

  readonly fmtBytes = fmtBytes;
  readonly fmtDuration = fmtDuration;

  acceptedFormats(): string {
    const h = this.health.health();
    if (!h) return '';
    return [...h.settings.video_extensions, ...h.settings.transcript_extensions].join(' ');
  }

  youtubeHint(): string {
    return this.health.health()?.settings.youtube_cookies_configured
      ? 'Cookies are configured, so downloads from this server should work.'
      : "No cookies configured. YouTube usually refuses downloads from a server IP with a "
        + 'bot check even though listing works — set BOOKFORGE_YOUTUBE_COOKIES_FILE, or use local files instead.';
  }

  setTab(tab: SourceTab): void {
    this.form.setTab(tab);
  }

  openFileBrowser(): void {
    this.fileInput().nativeElement.click();
  }

  onFileInputChange(event: Event): void {
    const input = event.target as HTMLInputElement;
    if (input.files?.length) this.form.addFiles(input.files);
    input.value = '';
  }

  onDrop(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
    if (event.dataTransfer?.files.length) this.form.addFiles(event.dataTransfer.files);
  }

  onDragOver(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(true);
  }

  onDragLeave(event: DragEvent): void {
    event.preventDefault();
    this.dragOver.set(false);
  }

  removeFile(index: number): void {
    this.form.removeFile(index);
  }

  async upload(): Promise<void> {
    this.uploadError.set(null);
    try {
      const result = await this.form.upload();
      if (result.rejected.length) {
        this.uploadError.set(`Skipped unsupported files: ${result.rejected.join(', ')}`);
      }
      await this.form.runPreview().catch(() => { /* surfaced via form.previewError */ });
    } catch (err) {
      this.uploadError.set(err instanceof Error ? err.message : String(err));
    }
  }

  async preview(): Promise<void> {
    await this.form.runPreview().catch(() => { /* surfaced via form.previewError */ });
  }

  onFolderPathChange(event: Event): void {
    this.form.folderPath.set((event.target as HTMLInputElement).value);
    this.form.clearPreview();
  }

  onYoutubeUrlChange(event: Event): void {
    this.form.youtubeUrl.set((event.target as HTMLInputElement).value);
    this.form.clearPreview();
  }
}
