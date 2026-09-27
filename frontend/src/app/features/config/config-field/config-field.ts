import { Component, input, output } from '@angular/core';
import { ConfigField } from '../../../core/models/api.models';

/** Renders one settings field from its schema (see bookforge.api.config_schema)
 *  and emits every change — it owns no state of its own, so the same field
 *  list drives both the API and this form with no risk of drifting apart. */
@Component({
  selector: 'app-config-field',
  templateUrl: './config-field.html',
})
export class ConfigFieldComponent {
  readonly field = input.required<ConfigField>();
  readonly value = input<unknown>();
  readonly valueChange = output<unknown>();

  asString(): string {
    const v = this.value();
    return v === undefined || v === null ? '' : String(v);
  }

  asBool(): boolean {
    return Boolean(this.value());
  }

  onText(event: Event): void {
    this.valueChange.emit((event.target as HTMLInputElement | HTMLSelectElement).value);
  }

  onNumber(event: Event): void {
    const raw = (event.target as HTMLInputElement).value;
    this.valueChange.emit(raw === '' ? undefined : Number(raw));
  }

  onCheckbox(event: Event): void {
    this.valueChange.emit((event.target as HTMLInputElement).checked);
  }

  passwordPlaceholder(): string {
    const f = this.field();
    return f.is_set ? `saved: ${f.value}` : 'not set';
  }
}
