import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient, withInterceptors } from '@angular/common/http';
import { Component, inject } from '@angular/core';
import { catchError, throwError } from 'rxjs';
import { AccountComponent } from './app/features/account/account.component';
import { SessionState } from './app/core/api/session-state.service';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [AccountComponent],
  template: '<app-account />'
})
class AppComponent {}

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient(withInterceptors([(request, next) => {
    const session = inject(SessionState);
    // 业务 HTTP 请求带 HttpOnly Cookie，401 时销毁工作台并清理其内存状态。
    return next(request.clone({ withCredentials: true })).pipe(catchError(error => {
      if (error.status === 401) session.user.set(null);
      return throwError(() => error);
    }));
  }]))]
}).catch(err => console.error(err));
