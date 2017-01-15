#!/usr/bin/perl -w
use strict;

my $EXPLICIT_TERMINAL_REDUCE = 0;
my $maxopts = scalar @ARGV;
for (my $i = 0; $i < $maxopts; $i++) {
  if ($ARGV[0] =~ /^--/) {
    my $opt = shift @ARGV;
    if ($opt eq '--help') {
      print <<EOT;
Usage: $0 < files.ptb

Converts parse trees in sexpr format (one tree per line) into
generate-reduce programs. Use --explicit_terminal_reduce to
add explicit REDUCE operations after terminal symbols.

Example input:
(S (NP (DT the) (NNS cat)) (VP (VBZ meows) (RB loudly)))

EOT
      exit(1);
    } elsif ($opt eq '--explicit_terminal_reduce') { $EXPLICIT_TERMINAL_REDUCE = 1; }
  }
}

while(<>) {
  print "# $_";
  chomp;
  my @toks = split /\s+/;
  my @terms = ();
  my @ops = ();
  for my $tok (@toks) {
    if ($tok =~ /^\((.+)$/) {
      die "Malformed input (extra paren): $_" if $tok =~ /\)$/;
      push @ops, "NT($1)";
    } else {
      my $term = $tok;
      $term =~ s/\)+$//;
      push @ops, "SHIFT($term)";
      push @terms, $term;
      while($tok =~ /\)/g) { push @ops, "REDUCE"; }
    }
  }
  print "@terms\n";
  for my $op (@ops) { print "$op\n"; }
  print "\n";
}

