from numpy import argmax

# Works out multinomial coefficient nC(k1,k2,...,km)
def multiCoeff(n, kList):
	kMaxFirstPos = argmax(kList)
	kMax = kList[kMaxFirstPos]
	coefficient = 1
	i = kMax + 1
	# Perform repeated multiplication and division 
	# for part of kList up to the maximum
	for k in kList[:kMaxFirstPos]:
		for j in range(1, k + 1):
			coefficient *= i
			coefficient //= j
			i += 1
	# Perform repeated multiplication and division 
	# for part of kList after the maximum
	for k in kList[kMaxFirstPos + 1:]:
		for j in range(1, k + 1):
			coefficient *= i
			coefficient //= j
			i += 1
	return coefficient

# Works out coefficients of multinomial expansion 
# given polynomial coefficents and power
def multiExpand(coeffsList, powersList, power):
	l = len(coeffsList)
	if l == 1:
		return [coeffsList[0] ** power], \
			[powersList[0] * power]
	if power == 0:
		return [1], [0]
	if power == 1:
		return coeffsList, powersList
	newCoeffs = []
	newPowers = []
	kList = [power]
	for i in range(l - 1):
		kList.append(0)
	newCoeff, newPower = multiCoeff(power, kList), 0
	for i in range(l):
		newCoeff *= (coeffsList[i] ** kList[i])
		newPower += (powersList[i] * kList[i])
	newCoeffs.append(newCoeff)
	newPowers.append(newPower)
	# kList should now look like [power, 0, ... 0] 
	# (e.g. [5, 0, 0] for power = 5, l = 3)
	# Used to work out multinomial coefficients
	while kList[l-1] != power:
		# Update kList
		if kList[l-2] != 0:
			kList[l-2] -= 1
			kList[l-1] += 1
		else:
			i = l - 2
			while kList[i] == 0:
				i -= 1
			kList[i] -= 1
			kList[i+1] = kList[l-1] + 1
			kList[l-1] = 0
		# Update newCoeff, newPower
		newCoeff, newPower = \
			multiCoeff(power, kList), 0
		for i in range(l):
			newCoeff *= (coeffsList[i] ** kList[i])
			newPower += (powersList[i] * kList[i])
		ind = newPowers.index(newPower) \
			if newPower in newPowers else -1
		if ind > -1:
			newCoeffs[ind] += newCoeff
		else:
			newCoeffs.append(newCoeff)
			newPowers.append(newPower)
	# Remove terms with zero coefficients
	i = 0
	while i < len(newCoeffs):
		if newCoeffs[i] == 0:
			newCoeffs.pop(i)
			newPowers.pop(i)
		else:
			i += 1
	return newCoeffs, newPowers

# Works out Qsigma - Q^p
def QsMinQp(a, b, QpC, QpP, p):
	QpC.pop(0), QpP.pop(0)	   # First elements cancel
	minQpC = [-el for el in QpC]	# Take -Q^p coeffs
	# Merge middle term of Qsigma into QsigmaMinQp
	ind = QpP.index(p) if p in QpP else -1
	if ind > -1:
		minQpC[ind] += (a ** p)
	else:
		ind = 0
		while ind < len(QpP) and QpP[ind] > p:
			ind += 1
		if ind == len(QpP):
			minQpC.extend((a ** p, p))
		else:
			minQpC.insert(ind, a ** p)
			QpP.insert(ind, p)
	# Deal with last element of Qsigma
	ind = -1 if 0 in QpP else -2
	if ind > -2:
		minQpC[ind] += (b ** p)
	else:
		minQpC.extend((b ** p, 0))
	# Pop first terms if zero
	while minQpC[0] and minQpC[0] == 0:
		minQpC.pop(0), QpP.pop(0)
	# Pop last terms if zero
	while minQpC[-1] and QpP[-1] == 0:
		minQpC.pop(), QpP.pop()
	return minQpC, QpP


# Works out reduction of x^m in terms of powers of tau
# Coefficients containing powers of x capped at x^2
def xPowerExpand(a, b, power, rDict):
	p, r, tauCoeffs = power // 3, power % 3, []
	# No need to recalculate if this monomial
	# is already in the dictionary - just return
	if power in rDict:
		return rDict[power]
	# "Base case" of recursion
	elif p == 0:
		new = [0 for i in range(0, r)]
		new.append(1)
		tauCoeffs.append(new)
	else:
		# May be higher powers of x before reduction
		for i in range(p + 1):
			new = [0 for j in range(r)]
			tauCoeffs.append(new)
			for j in range(i + 1):
				# Coefficient of x^j on tau^(i-power)
				coeff = multiCoeff(p, [p-i, j, i-j]) \
					* ((-a)**j) * ((-b)**(i-j))
				tauCoeffs[i].append(coeff)
		# Working from coefficients on powers of tau
		# closest to zero (higher indices)
		i = p
		while len(tauCoeffs[i]) > 3 and i >= 0:
			# Working from coeffs on power of tau
			# that are higher powers of x
			j = i + r
			while j > 2:
				# Reduce this power of x by recursion
				nCoeffs = xPowerExpand(a, b, j, rDict)
				# Merge smaller expansion into main 
				# expansion to progress reduction
				for k in range(len(nCoeffs)):
					for l in range(len(nCoeffs[k])):
						ind = i + k + 1 - len(nCoeffs)
						if len(tauCoeffs[ind]) > l:
							tauCoeffs[ind][l] += \
								tauCoeffs[i][j] * \
								nCoeffs[k][l]
						else:
							tauCoeffs[ind].append \
								(tauCoeffs[i][j] * \
								nCoeffs[k][l])
				# Remove high power of x from coeff 
				# of tau term once reduction is done
				tauCoeffs[i].pop(j)
				j -= 1
			i -= 1
	rDict[power] = tauCoeffs
	return tauCoeffs

# Works out coefficients of polynomial in 
# tau = y^-2 after reduction (x^3 = y^2 - ax - b)
def reducePoly(a, b, cList, pList, p, rDict):
	# Rewrite given polynomial in terms of tau
	# Reduce each power of x in turn and 
	# combine into a single list of coefficients
	# Use first monomial's expansion as a base
	expn = xPowerExpand(a, b, pList[0], rDict)
	expnStr = expStr(expn, 0)
	print(f"Reduction of x^{pList[0]} = {expnStr}")
	# Make sure the correct multiple is used
	factor = cList[0]
	for i in range(0, len(expn)):
		expn[i] = [factor * el for el in expn[i]]
	# Then add multiples of other lists
	for i in range(1, len(pList)):
		factor = cList[i]
		tmp = xPowerExpand(a, b, pList[i], rDict)
		tmpStr = expStr(tmp, 0)
		print(f"Reduction of x^{pList[i]} = {tmpStr}")
		for j in range(len(tmp)):
			for k in range(len(tmp[j])):
				expn[j + len(expn) - len(tmp)][k] += \
					(tmp[j][k] * factor)
	return expn

def rXY(a, b, p):
	# reductionDict is a dictionary for tracking 
	# monomials whose reductions are already computed
	reductionDict = {}
	# Arrays to represent the polynomial Q(x)
	coeffs, powers = [1, a, b], [3, 1, 0]
	# Step 1: find Qsigma - Q^p
	QpC, QpP = multiExpand(coeffs, powers, p)
	QsMinQpC, QsMinQpP = QsMinQp(a, b, QpC, QpP, p)
	QsMinQpStr = polyStr(QsMinQpC, QsMinQpP)
	print(f"(Qsigma - Q^p) = {QsMinQpStr}\n")
	# Step 2: rewrite Qsigma - Q^p in terms of tau
	expansion = reducePoly(a, b, \
		QsMinQpC, QsMinQpP, p, reductionDict)
	return expansion

# Coefficients on Q(x) = x^3 + ax + b; prime = p
a = int(input("Please enter a value for (a) in " + \
	"x^3 + ax + b. "))
b = int(input("Please also enter a value for (b). "))
p = int(input("Finally, please enter a prime p " + \
	"to determine the field. "))

# Set up arrays to represent Q as a polynomial
coeffs, powers = [1, a, b], [3, 1, 0]
print("\nThe polynomial entered was " + \
	f"{polyStr(coeffs, powers)}.\n")

# Output results
rXY_str = expStr(rXY(a, b, p), p)
print(f"\nR(x,y) = {rXY_str}")