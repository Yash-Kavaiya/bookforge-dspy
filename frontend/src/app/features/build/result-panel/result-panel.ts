import { Component, inject, input } from '@angular/core';
import { JobDto } from '../../../core/models/api.models';
import { WorkspaceService } from '../../../core/services/workspace.service';

@Component({
  selector: 'app-result-panel',
  templateUrl: './result-panel.html',
})
export class ResultPanel {
  protected readonly workspaces = inject(WorkspaceService);

  readonly job = input<JobDto | null>(null);

  bookUrl(slug: string, inline = true): string {
    return this.workspaces.bookUrl(slug, inline);
  }

  fileUrl(slug: string, path: string): string {
    return this.workspaces.fileUrl(slug, path);
  }
}
