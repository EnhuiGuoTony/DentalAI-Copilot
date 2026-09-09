import { bootstrapApplication } from '@angular/platform-browser';
import { provideHttpClient } from '@angular/common/http';
import { Component } from '@angular/core';
import { WorkbenchComponent } from './app/features/workbench/workbench.component';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [WorkbenchComponent],
  template: '<app-workbench />'
})
class AppComponent {}

bootstrapApplication(AppComponent, {
  providers: [provideHttpClient()]
}).catch(err => console.error(err));

