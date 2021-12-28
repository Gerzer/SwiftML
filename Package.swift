// swift-tools-version:5.5

import PackageDescription

let package = Package(
	name: "SwiftML",
	platforms: [
		.macOS(.v11)
	],
	products: [
		.library(
			name: "SwiftML",
			targets: [
				"SwiftML"
			]
		)
	],
	targets: [
		.target(
			name: "SwiftML"
		)
	]
)
