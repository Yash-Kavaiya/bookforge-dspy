import { Component, computed, inject, input } from '@angular/core';
import { JobUsage } from '../../../core/models/api.models';
import { HealthService } from '../../../core/services/health.service';
import { fmtCost, fmtTokens, priceFor } from '../../../core/util/format';

interface UsageChip {
  role: string;
  model: string;
  calls: number;
  promptTokens: number;
  completionTokens: number;
  failures: number;
  costLabel: string;
}

@Component({
  selector: 'app-usage-panel',
  templateUrl: './usage-panel.html',
})
export class UsagePanel {
  private readonly health = inject(HealthService);

  readonly usage = input<JobUsage | null>(null);

  readonly fmtTokens = fmtTokens;

  protected readonly chips = computed<UsageChip[]>(() => {
    const usage = this.usage();
    if (!usage) return [];
    const models = this.modelsMap();
    return Object.entries(usage.by_role).map(([role, r]) => {
      const price = priceFor(models[role]);
      const costLabel = price
        ? fmtCost((r.prompt_tokens / 1e6) * price[0] + (r.completion_tokens / 1e6) * price[1])
        : '';
      return {
        role, model: models[role] ?? '', calls: r.calls,
        promptTokens: r.prompt_tokens, completionTokens: r.completion_tokens,
        failures: r.failures, costLabel,
      };
    });
  });

  protected readonly totalTokens = computed(() => {
    const usage = this.usage();
    return usage ? usage.total.prompt_tokens + usage.total.completion_tokens : 0;
  });

  protected readonly costLine = computed(() => {
    const usage = this.usage();
    if (!usage) return '';
    const models = this.modelsMap();
    let total = 0, anyPriced = false, anyUnpriced = false;
    for (const [role, r] of Object.entries(usage.by_role)) {
      const price = priceFor(models[role]);
      if (price) {
        anyPriced = true;
        total += (r.prompt_tokens / 1e6) * price[0] + (r.completion_tokens / 1e6) * price[1];
      } else {
        anyUnpriced = true;
      }
    }
    if (!anyPriced) return 'cost n/a for this model';
    return `~${fmtCost(total)} est.${anyUnpriced ? ' (partial — some roles unpriced)' : ''}`;
  });

  /** `by_role` keys are arbitrary role strings from the server, not
   *  guaranteed to be one of HealthResponse['models']'s literal keys. */
  private modelsMap(): Record<string, string> {
    return this.health.health()?.models ?? {};
  }
}
