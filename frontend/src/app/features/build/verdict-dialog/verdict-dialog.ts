import { Component, ElementRef, inject, signal, viewChild } from '@angular/core';
import { ChapterSummary } from '../../../core/models/api.models';
import { WorkspaceService } from '../../../core/services/workspace.service';

const SEVERITY_ORDER: Record<string, number> = { critical: 0, major: 1, minor: 2 };

@Component({
  selector: 'app-verdict-dialog',
  templateUrl: './verdict-dialog.html',
})
export class VerdictDialog {
  private readonly workspaces = inject(WorkspaceService);
  private readonly dialogEl = viewChild.required<ElementRef<HTMLDialogElement>>('dialog');

  protected readonly chapter = signal<ChapterSummary | null>(null);
  protected readonly loadError = signal<string | null>(null);

  protected sortedDefects() {
    const defects = this.chapter()?.verdict?.defects ?? [];
    return [...defects].sort((a, b) => SEVERITY_ORDER[a.severity] - SEVERITY_ORDER[b.severity]);
  }

  async open(slug: string, videoId: string): Promise<void> {
    if (!slug) return;
    this.loadError.set(null);
    try {
      const payload = await this.workspaces.chapters(slug);
      const chapter = payload.chapters.find((c) => c.video_id === videoId) ?? null;
      if (!chapter) return;
      this.chapter.set(chapter);
      this.dialogEl().nativeElement.showModal();
    } catch (err) {
      this.loadError.set(err instanceof Error ? err.message : String(err));
    }
  }

  close(): void {
    this.dialogEl().nativeElement.close();
  }
}
