import { Injectable, inject } from '@angular/core';
import { ApiService } from './api.service';
import { ConfigResponse, ConfigTestResult, ConfigUpdateResult } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class ConfigService {
  private readonly api = inject(ApiService);

  load(): Promise<ConfigResponse> {
    return this.api.get<ConfigResponse>('/api/config');
  }

  save(settings: Record<string, unknown>, env: Record<string, string>): Promise<ConfigUpdateResult> {
    return this.api.put<ConfigUpdateResult>('/api/config', { settings, env });
  }

  test(role: string, model?: string, apiKey?: string, apiBase?: string): Promise<ConfigTestResult> {
    return this.api.post<ConfigTestResult>('/api/config/test', {
      role,
      model: model || undefined,
      api_key: apiKey || undefined,
      api_base: apiBase || undefined,
    });
  }
}
