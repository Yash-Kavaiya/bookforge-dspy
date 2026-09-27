import { Injectable, inject, signal } from '@angular/core';
import { ApiService } from './api.service';
import { ChaptersResponse, WorkspaceSummary } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class WorkspaceService {
  private readonly api = inject(ApiService);

  readonly workspaces = signal<WorkspaceSummary[]>([]);

  async refresh(): Promise<void> {
    try {
      const payload = await this.api.get<{ workspaces: WorkspaceSummary[] }>('/api/workspaces');
      this.workspaces.set(payload.workspaces);
    } catch {
      // Health already surfaces server-reachability problems; stay quiet here.
    }
  }

  chapters(slug: string): Promise<ChaptersResponse> {
    return this.api.get<ChaptersResponse>(`/api/workspaces/${slug}/chapters`);
  }

  bookUrl(slug: string, inline = true): string {
    return `/api/workspaces/${slug}/book${inline ? '' : '?inline=false'}`;
  }

  fileUrl(slug: string, path: string): string {
    return `/api/workspaces/${slug}/file?path=${encodeURIComponent(path)}`;
  }
}
