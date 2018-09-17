import json
out = []

with open('example.txt') as f: lines = f.readlines()
used = []
done_sent = {'name':'', 'eatType':'', 'food':'', 'priceRange':'', 'area':'', 'familyFriendly':'', 'near':'', 'customerRating':''}
resultList = [[],[],[],[]]

for l in lines:
	l = l.replace('\n','')
	
	if l.startswith('Input: '):
		sent = l.replace('Input: ','').split(' ')
		tmp = ''
		tag = ''
		for s in sent:
			if s == 'customer': continue
			elif s == 'rating': 
				if tag!='':
					done_sent[tag] = tmp
					tmp = ''
				tag = 'customerRating'
			elif s in done_sent: 
				if tag!='': 
					done_sent[tag] = tmp
					tmp = ''
				tag = s
			else:
				if tmp!='': tmp+=('_'+s)
				else: tmp = s
		print(done_sent)
	elif l.startswith('First Layer Output: '):
		sent = l.replace('First Layer Output: ','').split(' ')
		for s in sent:
			used.append(s)
			resultList[0].append(['new',s,'n'])
		print(resultList[0])
	elif l.startswith('Second Layer Output: '):
		sent = l.replace('Second Layer Output: ','').split(' ')
		for s in sent:
			if s in used: resultList[1].append(['old',s,'n'])
			else:
				used.append(s)
				resultList[1].append(['new',s,'v'])
		print(resultList[1])
	elif l.startswith('Third Layer Output: '):
		sent = l.replace('Third Layer Output: ','').split(' ')
		for s in sent:
			if s in used: resultList[2].append(['old',s,'n'])
			else:
				used.append(s)
				resultList[2].append(['new',s,'a'])
		print(resultList[2])
	elif l.startswith('Fourth Layer Output: '):
		sent = l.replace('Fourth Layer Output: ','').split(' ')
		for s in sent:
			if s in used: resultList[3].append(['old',s,'n'])
			else:
				used.append(s)
				resultList[3].append(['new',s,'other'])

		print(resultList[3])
		print()
		out.append({'input':done_sent, 'output':resultList, 'output_sent': ' '.join(sent)})
		with open('out.json','w') as f: json.dump(out, f)

		used = []
		done_sent = {'name':'', 'eatType':'', 'food':'', 'priceRange':'', 'area':'', 'familyFriendly':'', 'near':'', 'customerRating':''}
		resultList = [[],[],[],[]]
