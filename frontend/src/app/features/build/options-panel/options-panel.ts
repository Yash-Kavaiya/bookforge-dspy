import { Component, effect, inject, signal } from '@angular/core';
import { BuildFormService } from '../../../core/services/build-form.service';
import { BuildStreamService } from '../../../core/services/build-stream.service';
import { HealthService } from '../../../core/services/health.service';

const THEMES = [
  { value: 'textbook', title: 'Textbook', hint: 'Chapter openers, margin notes, callouts, list of figures' },
  { value: 'academic', title: 'Academic', hint: 'Restrained headings, no margin column' },
  { value: 'compact', title: 'Compact', hint: '10pt, tighter, fewer front-matter pages' },
];

@Component({
  selector: 'app-options-panel',
  templateUrl: './options-panel.html',
})
export class OptionsPanel {
  protected readonly form = inject(BuildFormService);
  protected readonly stream = inject(BuildStreamService);
  protected readonly health = inject(HealthService);

  protected readonly themes = THEMES;
  protected readonly starting = signal(false);
  protected readonly startError = signal<string | null>(null);

  constructor() {
    // pdflatex missing means nothing can be verified by compilation. Say so
    // where the choice is made rather than leaving it to be discovered at the end.
    effect(() => {
      if (this.pdflatexMissing()) this.form.compileLatex.set(false);
    });
  }

  onTitleChange(event: Event): void {
    this.form.bookTitle.set((event.target as HTMLInputElement).value);
  }

  onThemeChange(value: string): void {
    this.form.theme.set(value);
  }

  onMaxVideosChange(event: Event): void {
    const raw = (event.target as HTMLInputElement).value;
    this.form.maxVideos.set(raw === '' ? null : Number(raw));
  }

  onQaIterationsChange(event: Event): void {
    const raw = (event.target as HTMLInputElement).value;
    this.form.qaIterations.set(raw === '' ? null : Number(raw));
  }

  onCompileLatexChange(event: Event): void {
    this.form.compileLatex.set((event.target as HTMLInputElement).checked);
  }

  pdflatexMissing(): boolean {
    const h = this.health.health();
    return !!h && !h.tools.pdflatex;
  }

  buildHint(): string {
    if (!this.form.preview()) return 'Preview a source first.';
    return `${this.form.preview()!.count} chapter(s) will be produced.`;
  }

  async startBuild(): Promise<void> {
    this.starting.set(true);
    this.startError.set(null);
    try {
      const job = await this.form.submitBuild();
      this.stream.start(job.id);
      document.getElementById('panel-progress')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch (err) {
      this.startError.set(err instanceof Error ? err.message : String(err));
    } finally {
      this.starting.set(false);
    }
  }
}
