# Subagents cannot invoke subagents

**Status:** platform limitation, established by direct test during
EVAL-AUTO. Recorded because it invalidates an obvious-looking design and
would otherwise be rediscovered from scratch.

## What was attempted

Mass generation of the known-item eval set needs ~500 calls to the
`eval-query-generator` subagent. The natural design is a fan-out: the
main loop dispatches ~20 orchestrator agents, each responsible for a
batch of 25 songs, and each orchestrator calls the generator once per
song and writes the replies to disk. The main loop then spends 20 calls
instead of 500 and never sees a lyrics fragment.

## What happened

The first orchestrator returned without writing anything:

> the instructions require me to invoke a subagent called
> `eval-query-generator` using a "Task tool", but this tool is not
> available in my current execution environment

It listed what it did have -- file operations, Bash, web tools -- and the
`Task` tool was absent. This is not a permissions setting on the agent
definition and not a consequence of the `tools:` frontmatter, which in
this case granted the orchestrator everything. **An agent running as a
subagent has no `Task` tool at all.** Agent invocation is available only
from the main conversation loop.

## Consequences

Every subagent call is issued from the main loop, and therefore:

- **Fan-out does not reduce main-loop cost.** The cost of N generations
  is N dispatches plus N results in the main context, regardless of how
  the work is grouped. The only lever left is how many songs one call
  covers, which is why batched generation had to be evaluated at all
  (see the batching experiment in the EVAL-AUTO report).
- **Anything a subagent must produce in bulk has to be either written by
  that subagent itself or returned through the main loop.** There is no
  intermediate tier.
- **Prompt material passes through the main context.** The fan-out design
  would have kept copyrighted fragments out of the main transcript
  entirely; without it, that property cannot be had, and the mitigation
  is to keep fragments out of *reports* rather than out of dispatch.

## What this does not mean

Subagents remain useful for the thing they are good at: isolating a task
so that a large amount of intermediate reasoning, file reading and trial
and error never enters the main context, and only a conclusion comes
back. That property is unaffected. What fails is specifically using them
as a tree to multiply throughput.
