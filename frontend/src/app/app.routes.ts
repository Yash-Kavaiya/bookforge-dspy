import { Routes } from '@angular/router';

export const routes: Routes = [
  {
    path: '',
    loadComponent: () => import('./features/build/build-page/build-page').then((m) => m.BuildPage),
    title: 'BookForge — Build',
  },
  {
    path: 'config',
    loadComponent: () => import('./features/config/config-page/config-page').then((m) => m.ConfigPage),
    title: 'BookForge — Config',
  },
  { path: '**', redirectTo: '' },
];
