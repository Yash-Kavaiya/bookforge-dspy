import { HttpClient, HttpErrorResponse } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { firstValueFrom } from 'rxjs';

/** Thin HttpClient wrapper: every non-2xx becomes an Error whose message is
 *  the server's `detail` (FastAPI's HTTPException shape) rather than a
 *  generic "Http failure response for ..." — callers just read `err.message`. */
@Injectable({ providedIn: 'root' })
export class ApiService {
  private readonly http = inject(HttpClient);

  get<T>(path: string): Promise<T> {
    return this.run(this.http.get<T>(path));
  }

  post<T>(path: string, body?: unknown): Promise<T> {
    return this.run(this.http.post<T>(path, body ?? {}));
  }

  put<T>(path: string, body?: unknown): Promise<T> {
    return this.run(this.http.put<T>(path, body ?? {}));
  }

  postForm<T>(path: string, form: FormData): Promise<T> {
    return this.run(this.http.post<T>(path, form));
  }

  private async run<T>(source: ReturnType<HttpClient['get']>): Promise<T> {
    try {
      return (await firstValueFrom(source)) as T;
    } catch (err) {
      if (err instanceof HttpErrorResponse) {
        const detail = (err.error && typeof err.error === 'object' && 'detail' in err.error)
          ? String((err.error as { detail: unknown }).detail)
          : err.message;
        throw new Error(detail || `HTTP ${err.status}`);
      }
      throw err;
    }
  }
}
