import { CommonModule } from '@angular/common';
import { Component, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';
import { DentalAiApiService } from '../../core/api/dentalai-api.service';
import { ChatMessage } from '../../core/models/api.models';

@Component({
  selector: 'app-workbench',
  standalone: true,
  imports: [CommonModule, FormsModule],
  templateUrl: './workbench.component.html',
  styleUrl: './workbench.component.scss'
})
export class WorkbenchComponent {
  readonly messages = signal<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'I am ready. Send a message to test the LLM integration.'
    }
  ]);
  readonly input = signal('Explain this project in one minute for an AI Engineer interview.');
  readonly status = signal('Ready');
  readonly provider = signal('not connected');
  readonly loading = signal(false);

  constructor(private readonly api: DentalAiApiService) {}

  send(): void {
    const message = this.input().trim();
    if (!message || this.loading()) return;

    const history = this.messages();
    this.messages.set([...history, { role: 'user', content: message }]);
    this.input.set('');
    this.loading.set(true);
    this.status.set('Waiting for LLM...');

    this.api.chat(message, history)
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: response => {
          this.provider.set(`${response.provider}${response.mock ? ' (mock)' : ''}`);
          this.messages.set([...this.messages(), { role: 'assistant', content: response.reply }]);
          this.status.set('Response received');
        },
        error: error => {
          this.messages.set([
            ...this.messages(),
            {
              role: 'assistant',
              content: `Request failed: ${error?.error?.detail ?? error.message ?? 'Unknown error'}`
            }
          ]);
          this.status.set('Request failed');
        }
      });
  }

  clear(): void {
    this.messages.set([]);
    this.status.set('Ready');
  }
}
