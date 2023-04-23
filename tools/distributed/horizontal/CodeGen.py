import re

PLACEHOLDER = re.compile('\\$([^\\$]+)\\$')

class Snippet:
	last = None
	def __init__(self, owner, text, postfix):
		self.owner = owner
		if self.owner.last is not None:
			with self.owner.last:
				pass
		self.owner.write("".join(text))
		self.owner.last = self
		self.postfix = postfix
		
	def __enter__(self):
		self.owner.write("{")
		self.owner.current_indent += 1
		self.owner.last = None
		
	def __exit__(self, a, b, c):
		if self.owner.last is not None:
			with self.owner.last:
				pass
		self.owner.current_indent -= 1
		self.owner.write("}" + self.postfix)
		
class Subs:
	def __init__(self, owner, subs):
		self.owner = owner
		self.subs = subs
		
	def __enter__(self):
		self.owner.substack = [self.subs] + self.owner.substack
		
	def __exit__(self, a, b, c):
		self.owner.substack = self.owner.substack[1:]
		

class CodeFile:
	def __init__(self, filename):
		self.current_indent = 0
		self.last = None
		self.out = open(filename,"w")
		self.indent = "\t"
		self.substack = []
		
	def close(self):
		self.out.close()
		self.out = None
	
	def write(self, x, indent=0):
		self.out.write(self.indent * (self.current_indent+indent) + x + "\n")
		
	def format(self, text):
		while True:
			m = PLACEHOLDER.search(text)
			if m is None:
				return text
			s = None
			for sub in self.substack:
				if m.group(1) in sub:
					s = sub[m.group(1)]
					break
			if s is None:
				raise Exception("Substitution '%s' not set." % m.groups(1))
			text = text[:m.start()] + str(s) + text[m.end():]		
		
	def subs(self, **subs):
		return Subs(self, subs)
		
	def __call__(self, text):
		self.write(self.format(text))
		
	def block(self, text, postfix=""):
		return Snippet(self, self.format(text), postfix)

class CppFile(CodeFile):
	def __init__(self, filename):
		CodeFile.__init__(self, filename)
		
	def label(self, text):
		self.write(self.format(text) + ":", -1)
		
__all__ = [ "CppFile", "CodeFile" ]
