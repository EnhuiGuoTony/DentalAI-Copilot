import { Component, OnInit, inject, signal } from '@angular/core';
import { CommonModule } from '@angular/common';
import { FormsModule } from '@angular/forms';
import { finalize } from 'rxjs';
import { DentalAiApiService } from '../../core/api/dentalai-api.service';
import { SessionState } from '../../core/api/session-state.service';
import { WorkbenchComponent } from '../workbench/workbench.component';

/** 认证门禁覆盖整个工作台；服务端也会独立验证每个业务请求。 */
@Component({
  selector: 'app-account', standalone: true, imports: [CommonModule, FormsModule, WorkbenchComponent],
  templateUrl: './account.component.html',
  styles: [`:host{display:block} .account{max-width:600px;margin:2rem auto;padding:2rem;background:white;border:1px solid #dce5ef;border-radius:16px}
    form{display:grid;gap:12px}label{display:grid;gap:6px}input,button{padding:10px;border:1px solid #bccddd;border-radius:8px}
    .bar{padding:12px 24px;background:#123650;color:white;display:flex;gap:16px;align-items:center;flex-wrap:wrap}
    .bar span{flex:1}.error{color:#b42318}.notice{color:#236848}`]
})
export class AccountComponent implements OnInit {
  readonly session = inject(SessionState);
  private readonly api = inject(DentalAiApiService);
  readonly ready = signal(false);
  readonly busy = signal(false);
  readonly error = signal('');
  readonly notice = signal('');
  registerMode = false;
  showProfile = false;
  username = ''; password = ''; displayName = ''; currentPassword = ''; newPassword = '';

  ngOnInit(): void {
    this.api.me().pipe(finalize(() => this.ready.set(true))).subscribe({
      next: user => { this.session.user.set(user); this.displayName = user.display_name; }, error: () => {}
    });
  }
  submit(): void {
    if (this.busy()) return;
    this.busy.set(true); this.error.set(''); this.notice.set('');
    const request = this.registerMode ? this.api.register(this.username, this.password, this.displayName) : this.api.login(this.username, this.password);
    request.pipe(finalize(() => this.busy.set(false))).subscribe({
      next: user => { this.session.user.set(user); this.password = ''; this.displayName = user.display_name; },
      error: e => this.error.set(e.status === 409 ? '用户名已存在。' : '登录或注册失败，请检查账号与密码（至少 10 位）。')
    });
  }
  saveProfile(): void {
    if (this.busy()) return;
    this.busy.set(true); this.error.set('');
    this.api.updateProfile(this.displayName).pipe(finalize(() => this.busy.set(false))).subscribe({
      next: user => { this.session.user.set(user); this.notice.set('资料已保存。'); }, error: () => this.error.set('保存失败。')
    });
  }
  changePassword(): void {
    if (this.busy()) return;
    this.busy.set(true); this.error.set('');
    this.api.password(this.currentPassword, this.newPassword).pipe(finalize(() => this.busy.set(false))).subscribe({
      next: () => { this.session.user.set(null); this.currentPassword = this.newPassword = ''; this.showProfile = false; this.notice.set('密码已修改，请重新登录。'); },
      error: () => this.error.set('修改失败，请检查当前密码及新密码长度。')
    });
  }
  logout(): void {
    this.api.logout().subscribe({ next: () => { this.session.user.set(null); this.showProfile = false; this.error.set(''); }, error: () => this.error.set('退出失败，请重试。') });
  }
}
