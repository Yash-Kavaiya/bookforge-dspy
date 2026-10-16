import { Component, OnInit, inject } from '@angular/core';
import { WorkspaceSummary } from '../../../core/models/api.models';
import { BuildStreamService } from '../../../core/services/build-stream.service';
import { WorkspaceService } from '../../../core/services/workspace.service';

@Component({
  selector: 'app-library-panel',
  templateUrl: './library-panel.html',
})
export class LibraryPanel implements OnInit {
  protected readonly workspaces = inject(WorkspaceService);
  private readonly stream = inject(BuildStreamService);

  ngOnInit(): void {
    void this.workspaces.refresh();
  }

  refresh(): void {
    void this.workspaces.refresh();
  }

  bookUrl(ws: WorkspaceSummary): string {
    return this.workspaces.bookUrl(ws.slug);
  }

  async inspect(ws: WorkspaceSummary): Promise<void> {
    try {
      const payload = await this.workspaces.chapters(ws.slug);
      this.stream.showChapters(ws.slug, payload.chapters);
      document.getElementById('panel-progress')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    } catch {
      /* transient; the panel just won't update */
    }
  }
}
