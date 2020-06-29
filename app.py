import cv2
import string
from data_loader import DataLoader
import pygame
import time
import os

class App:

	def __init__(self, model = None, check_accuracy= False):
		self.char_list = string.ascii_letters

		if check_accuracy and model != None:
			self.dl = DataLoader()
			self.my_model = model
			self.accuracy = self.update_accuracy()
			print("accuracy=", self.accuracy)
		else:
			dl = DataLoader()
			self.WHITE = (255,255,255)
			self.GREY = (247,241,241)
			self.BLACK = (0,0,0)
			pygame.init()
			dims = (640,480)
			self.screen = pygame.display.set_mode(dims)
			pygame.display.set_caption('Handwriting Recognition')
			running = True
			self.screen.fill((self.GREY))
			pygame.draw.rect(self.screen, (0,0,0), (30,50,570,300), 2)
			self.screen.fill(self.WHITE, rect=(31,51,569,299))
			clock = pygame.time.Clock()
			p1 = None
			button = self.make_button("GUESS",(100,370,70,30))
			clear_button = self.make_button("CLEAR",(200,370, 70,30))


			while running:
				running, p1 = self.action(p1, button,clear_button, dl, model)
				clock.tick(60)

	def action(self, p1, button,clear_button, dl, model):
		left_pressed, middle_pressed, right_pressed = pygame.mouse.get_pressed()
		for event in pygame.event.get():
			if event.type == pygame.QUIT:
				return [False,None]
			elif left_pressed:
				if self.valid(pygame.mouse.get_pos()):
					#draw line save last pos and draw line in btwn
					self.draw_line(p1,pygame.mouse.get_pos())
					p1 = pygame.mouse.get_pos()
				elif button.collidepoint(pygame.mouse.get_pos()):
					print("BUTTON PRESSED")
					self.guess(dl, model)
				elif clear_button.collidepoint(pygame.mouse.get_pos()):
					self.screen.fill((255,255,255), rect=(31,51,569,299))
			if not left_pressed:
				p1 = None
		pygame.display.update()
		return [True, p1]

	def make_button(self, btn_name,dims):
		button = pygame.Rect(dims)
		pygame.draw.rect(self.screen, (0,0,0), (dims), 2)
		self.screen.fill(self.WHITE, rect=(dims[0]+1, dims[1]+1, dims[2]-1, dims[3]-1))

		myfont = pygame.font.SysFont('calibri', 17)
		myfont.set_bold(True)
		textsurface = myfont.render(btn_name, False, (0, 0, 0))
		self.screen.blit(textsurface,(dims[0]+8, dims[1]+8))
		return button


	def guess(self, dl, model):
		save_file = "image_guess/{}.png".format(time.time())
		pygame.image.save(self.screen, save_file)
		dl.image_crop(save_file, (31,599,51,350))
		
		if model != None:
			img = cv2.imread(save_file,cv2.IMREAD_GRAYSCALE)
			img = dl.image_extract(img)
			#cv2.imshow('y',img)
			#cv2.waitKey(0)
			model.my_predict([img], self.char_list)


	def valid(self, mouse_pos):
		x, y = mouse_pos
		if x < 600 and x > 30 and y > 50 and y < 350:
			return True

		return False

	def draw_line(self, p1, p2):
		if p1 == None:
			pygame.draw.lines(self.screen, self.BLACK, False, [(p2),(p2)], 8)
		else:
			pygame.draw.lines(self.screen, self.BLACK, False, [(p1),(p2)], 8)

	def update_accuracy(self):
		PATH = "image_guess"
		correct = 0.0
		total = 0.0
		for filename in os.listdir(PATH):
			if '-' in filename:
				lis = filename.split('-')
				#print("actual=",lis[0])
				img = cv2.imread(os.path.join(PATH,filename),cv2.IMREAD_GRAYSCALE)
				img = self.dl.image_extract(img)
				out, prediction = self.my_model.my_predict([img], self.char_list)
				if lis[0] == prediction:
					correct += 1
					#print("actual=",lis[0])
					#print("predicted=",prediction)
				elif len(lis[0]) == self.lcs(lis[0],prediction):
					correct +=1
					#print("actual=",lis[0])
					#print("predicted=",prediction)
				total += 1
				print("actual=",lis[0])
				print("predicted=",prediction)
		print("correct=",correct)
		print("total=",total)
		return  (correct//float(total))

	def lcs(self,X , Y): 
	    m = len(X) 
	    n = len(Y) 	  
	    L = [[None]*(n+1) for i in range(m+1)] 
	    for i in range(m+1): 
	        for j in range(n+1): 
	            if i == 0 or j == 0 : 
	                L[i][j] = 0
	            elif X[i-1] == Y[j-1]: 
	                L[i][j] = L[i-1][j-1]+1
	            else: 
	                L[i][j] = max(L[i-1][j] , L[i][j-1]) 	  
	    return L[m][n]

if __name__ == '__main__':
	test = App()
	print(test.lcs("hello","hell"))