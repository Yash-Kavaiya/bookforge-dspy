import { Injectable, inject, signal } from '@angular/core';
import { ApiService } from './api.service';
import { HealthResponse } from '../models/api.models';

@Injectable({ providedIn: 'root' })
export class HealthService {
  private readonly api = inject(ApiService);

  readonly health = signal<HealthResponse | null>(null);
  readonly error = signal<string | null>(null);

  async refresh(): Promise<HealthResponse | null> {
    try {
      const health = await this.api.get<HealthResponse>('/api/health');
      this.health.set(health);
      this.error.set(null);
      return health;
    } catch (err) {
      this.error.set(err instanceof Error ? err.message : String(err));
      return null;
    }
  }
}
