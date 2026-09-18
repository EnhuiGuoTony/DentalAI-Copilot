import { Injectable, signal } from '@angular/core';
import { UserProfile } from '../models/api.models';

/** 只保存可展示的账号资料；认证 Cookie 由浏览器管理，脚本不能读取。 */
@Injectable({ providedIn: 'root' })
export class SessionState {
  readonly user = signal<UserProfile | null>(null);
}
