import { Component, input, output } from '@angular/core';
import { ChapterSummary } from '../../../core/models/api.models';

@Component({
  selector: 'app-chapter-card',
  templateUrl: './chapter-card.html',
})
export class ChapterCard {
  readonly chapter = input.required<ChapterSummary>();
  readonly select = output<ChapterSummary>();

  score(): number {
    return this.chapter().qa_score ?? 0;
  }

  scorePercent(): number {
    return Math.round(this.score() * 100);
  }

  chapterNumber(): string {
    return String(this.chapter().chapter_number).padStart(2, '0');
  }
}
