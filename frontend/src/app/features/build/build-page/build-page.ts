import { Component, OnInit, inject } from '@angular/core';
import { SourcePanel } from '../source-panel/source-panel';
import { OptionsPanel } from '../options-panel/options-panel';
import { ProgressPanel } from '../progress-panel/progress-panel';
import { LibraryPanel } from '../library-panel/library-panel';
import { BuildStreamService } from '../../../core/services/build-stream.service';

@Component({
  selector: 'app-build-page',
  imports: [SourcePanel, OptionsPanel, ProgressPanel, LibraryPanel],
  templateUrl: './build-page.html',
})
export class BuildPage implements OnInit {
  private readonly stream = inject(BuildStreamService);

  ngOnInit(): void {
    // A page reload during a long run should pick up where it left off
    // instead of appearing idle.
    void this.stream.reattachIfRunning();
  }
}
