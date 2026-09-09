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
  private readonly maxHistoryMessages = 6;
  private readonly maxMessageChars = 4000;
  private conversationHistory: ChatMessage[] = [];

  // Prompts are retained only in this small context window; only AI replies render.
  readonly messages = signal<ChatMessage[]>([]);
  readonly input = signal('Explain this project in one minute for an AI Engineer interview.');
  readonly connectionLabel = signal('Connect model');
  readonly connected = signal(false);
  readonly connecting = signal(false);
  readonly loading = signal(false);

  constructor(private readonly api: DentalAiApiService) {}

  connect(): void {
    if (this.connecting() || this.connected()) return;

    this.connecting.set(true);
    this.connectionLabel.set('Connecting...');
    this.api.connectChat()
      .pipe(finalize(() => this.connecting.set(false)))
      .subscribe({
        next: response => {
          this.connected.set(response.connected);
          this.connectionLabel.set(response.connected ? 'Model connected' : 'Connection unavailable');
        },
        error: () => this.connectionLabel.set('Connection unavailable')
      });
  }

  send(): void {
    const message = this.input().trim();
    if (!message || this.loading() || !this.connected()) return;

    const history = this.conversationHistory.slice(-this.maxHistoryMessages);
    const userTurn: ChatMessage = {
      role: 'user',
      content: message.slice(0, this.maxMessageChars)
    };
    this.input.set('');
    this.loading.set(true);

    this.api.chat(userTurn.content, history)
      .pipe(finalize(() => this.loading.set(false)))
      .subscribe({
        next: response => {
          const assistantTurn: ChatMessage = { role: 'assistant', content: response.reply };
          this.conversationHistory = [...this.conversationHistory, userTurn, assistantTurn]
            .slice(-this.maxHistoryMessages);
          this.messages.set([...this.messages(), assistantTurn]);
        },
        error: () => {
          this.connected.set(false);
          this.connectionLabel.set('Request failed — reconnect');
        }
      });
  }

  clear(): void {
    this.messages.set([]);
    this.conversationHistory = [];
  }
}
