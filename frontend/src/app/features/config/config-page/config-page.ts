import { Component, OnInit, computed, inject, signal } from '@angular/core';
import { ConfigService } from '../../../core/services/config.service';
import { ConfigField, ConfigGroup, ConfigTestResult } from '../../../core/models/api.models';
import { ConfigFieldComponent } from '../config-field/config-field';

const PROVIDER_ENV_HINTS: Record<string, string> = {
  GOOGLE_API_KEY: 'Gemini — pair with a gemini/... model id',
  ANTHROPIC_API_KEY: 'Anthropic — pair with an anthropic/... model id',
  NVIDIA_API_KEY: 'NVIDIA NIM, read directly by some SDK paths',
  OPENAI_API_KEY: 'Real OpenAI — pair with an openai/... model id and no API base',
};

const TEST_ROLES = ['analyst', 'writer', 'critic', 'caption', 'reflection'] as const;

@Component({
  selector: 'app-config-page',
  imports: [ConfigFieldComponent],
  templateUrl: './config-page.html',
})
export class ConfigPage implements OnInit {
  private readonly configService = inject(ConfigService);

  protected readonly testRoles = TEST_ROLES;
  protected readonly providerEnvKeys = Object.keys(PROVIDER_ENV_HINTS);
  protected readonly providerEnvHints = PROVIDER_ENV_HINTS;

  protected readonly fields = signal<ConfigField[]>([]);
  protected readonly groups = signal<ConfigGroup[]>([]);
  protected readonly providerEnv = signal<Record<string, boolean>>({});
  protected readonly envFile = signal<string | null>(null);

  /** Current form values, keyed by field.key. Secret fields start blank —
   *  never round-tripping the raw key back into the DOM is the whole point
   *  of masking it server-side. */
  protected readonly formValues = signal<Record<string, unknown>>({});
  protected readonly envValues = signal<Record<string, string>>({});

  protected readonly loadError = signal<string | null>(null);
  protected readonly saving = signal(false);
  protected readonly saveMessage = signal<{ text: string; ok: boolean } | null>(null);

  protected readonly testRole = signal<string>('analyst');
  protected readonly testing = signal(false);
  protected readonly testResult = signal<ConfigTestResult | null>(null);

  protected readonly fieldsByGroup = computed(() => {
    const byGroup = new Map<string, ConfigField[]>();
    for (const field of this.fields()) {
      const list = byGroup.get(field.group) ?? [];
      list.push(field);
      byGroup.set(field.group, list);
    }
    return byGroup;
  });

  ngOnInit(): void {
    void this.load();
  }

  async load(): Promise<void> {
    try {
      const data = await this.configService.load();
      this.fields.set(data.fields);
      this.groups.set(data.groups);
      this.providerEnv.set(data.provider_env);
      this.envFile.set(data.env_file);
      this.loadError.set(null);

      const values: Record<string, unknown> = {};
      for (const field of data.fields) values[field.key] = field.secret ? '' : field.value;
      this.formValues.set(values);
      this.envValues.set({});
    } catch (err) {
      this.loadError.set(err instanceof Error ? err.message : String(err));
    }
  }

  onFieldChange(key: string, value: unknown): void {
    this.formValues.update((values) => ({ ...values, [key]: value }));
  }

  onEnvChange(name: string, event: Event): void {
    const value = (event.target as HTMLInputElement).value;
    this.envValues.update((values) => ({ ...values, [name]: value }));
  }

  envPlaceholder(name: string): string {
    return this.providerEnv()[name] ? 'configured' : 'not set';
  }

  async save(): Promise<void> {
    const settings: Record<string, unknown> = {};
    for (const field of this.fields()) {
      const value = this.formValues()[field.key];
      if (field.secret) {
        if (value !== '' && value !== undefined) settings[field.key] = value;
        continue;
      }
      if (value !== undefined) settings[field.key] = value;
    }
    const env: Record<string, string> = {};
    for (const [name, value] of Object.entries(this.envValues())) {
      if (value) env[name] = value;
    }

    this.saving.set(true);
    this.saveMessage.set(null);
    try {
      const result = await this.configService.save(settings, env);
      const count = result.applied.length + result.env_updated.length;
      this.saveMessage.set({
        ok: true,
        text: count
          ? `Saved ${count} value(s) to .env and applied them immediately — no restart needed.`
          : 'Nothing changed.',
      });
      await this.load();
    } catch (err) {
      this.saveMessage.set({ ok: false, text: `Save failed: ${err instanceof Error ? err.message : String(err)}` });
    } finally {
      this.saving.set(false);
    }
  }

  async testConnection(): Promise<void> {
    const role = this.testRole();
    const model = this.formValues()[`${role}_model`] as string | undefined;
    const apiKey = this.formValues()['openai_api_key'] as string | undefined;
    const apiBase = this.formValues()['openai_api_base'] as string | undefined;

    this.testing.set(true);
    this.testResult.set(null);
    try {
      const result = await this.configService.test(role, model, apiKey, apiBase);
      this.testResult.set(result);
    } catch (err) {
      this.testResult.set({ ok: false, model: model ?? '', latency_ms: 0, error: err instanceof Error ? err.message : String(err) });
    } finally {
      this.testing.set(false);
    }
  }
}
