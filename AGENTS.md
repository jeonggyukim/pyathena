# On-the-fly analysis project coordination

For this project's readers and discovery work, read /home/sm69/athena_radial_profile/AGENTS.md, /home/sm69/athena_radial_profile/PLANS.md, and /home/sm69/athena_radial_profile/plans/onthefly-analysis-master.md before planning or implementation.

Read the single shared output-time specification at /home/sm69/athena_radial_profile/OUTPUT_CONTRACT.md when changing discovery, output reading, or numbering. Base pyathena retains native file numbering; canonical application indexing belongs in core_formation. Do not duplicate the contract here or assume its approved target is already implemented. A revision requires reviewing Athena++, consumers, tests, and existing-output compatibility together.

Every future project planning session must update the master plan and its dedicated smaller plan; every implementation milestone must update both. Preserve unrelated user edits. Use the pyathena Python environment, place all new simulation/analysis artifacts only under /scratch/gpfs/sm69/onthefly-rprof-test, and include Co-authored-by: Codex <codex@openai.com> in focused commits. Update explanatory PDFs after major milestones.
