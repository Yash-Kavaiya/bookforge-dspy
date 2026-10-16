import { AfterViewChecked, Component, ElementRef, input, viewChild } from '@angular/core';
import { JobEventDto } from '../../../core/models/api.models';
import { fmtClock } from '../../../core/util/format';

const KIND_ICON: Record<string, string> = { stage: '▸', info: '·', warning: '!', error: '✕', done: '✓' };

@Component({
  selector: 'app-log-panel',
  templateUrl: './log-panel.html',
})
export class LogPanel implements AfterViewChecked {
  readonly events = input<JobEventDto[]>([]);
  private readonly logEl = viewChild<ElementRef<HTMLDivElement>>('log');

  readonly fmtClock = fmtClock;
  readonly iconFor = (kind: string) => KIND_ICON[kind] ?? '·';

  private lastRenderedCount = 0;

  // Only autoscroll if the user was already at the bottom — yanking the view
  // while someone is reading earlier output is hostile.
  ngAfterViewChecked(): void {
    const count = this.events().length;
    if (count === this.lastRenderedCount) return;
    this.lastRenderedCount = count;
    const el = this.logEl()?.nativeElement;
    if (!el) return;
    const wasAtBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 80;
    if (wasAtBottom) el.scrollTop = el.scrollHeight;
  }
}
