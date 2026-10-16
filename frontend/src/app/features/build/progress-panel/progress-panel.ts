import { Component, ElementRef, inject, viewChild } from '@angular/core';
import { BuildStreamService } from '../../../core/services/build-stream.service';
import { ChapterSummary } from '../../../core/models/api.models';
import { ChapterCard } from '../chapter-card/chapter-card';
import { UsagePanel } from '../usage-panel/usage-panel';
import { LogPanel } from '../log-panel/log-panel';
import { ResultPanel } from '../result-panel/result-panel';
import { VerdictDialog } from '../verdict-dialog/verdict-dialog';
import { HealthService } from '../../../core/services/health.service';

@Component({
  selector: 'app-progress-panel',
  imports: [ChapterCard, UsagePanel, LogPanel, ResultPanel, VerdictDialog],
  templateUrl: './progress-panel.html',
})
export class ProgressPanel {
  protected readonly stream = inject(BuildStreamService);
  protected readonly health = inject(HealthService);
  private readonly verdictDialog = viewChild.required(VerdictDialog);

  cancel(): void {
    void this.stream.cancel();
  }

  onChapterSelect(chapter: ChapterSummary): void {
    const slug = this.stream.channelSlug();
    if (!slug) return;
    void this.verdictDialog().open(slug, chapter.video_id);
  }

  concurrencyNote(): string {
    const workers = this.health.health()?.concurrency.media_prefetch_workers;
    if (workers === undefined) return '';
    return `1 chapter at a time · ${workers} media-prefetch thread${workers === 1 ? '' : 's'}`;
  }
}
